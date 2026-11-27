"""
Train_Hybrid.py — обучение HybridPointTransformer (53 класса, 3DTeethSeg'22).

Патч от 2026-11-26. Что изменено и почему (см. DIAGNOSTIC.md):

  1. Восстановление после early stopping больше не "зацикливает" обучение.
     Раньше resume шёл из checkpoint_best.pth, а счётчик patience не
     восстанавливался => каждый перезапуск проигрывал одни и те же эпохи и
     снова упирался в early stopping ("обучение снова встало").

  2. Валидационная метрика считается по матрице ошибок (confusion), накопленной за
     всю эпоху, а не как среднее по образцам. Прежний вариант усреднял
     per-sample mIoU, а внутри образца - IoU только по присутствующим
     классам; для 53 классов с 65% фона такая оценка крайне шумная
     (sd = 0.044, размах 0.209-0.353 на вашем логе), и early stopping
     срабатывал раньше времени.

  3. Ранняя остановка теперь по EMA от сглаженной метрики, а patience
     восстанавливается из чекпоинта.

  4. Разбиение на train/val - по пациентам (case), а не по файлам.
     Раньше обе челюсти одного пациента могли попасть и в train, и в val.
     По умолчанию используются готовые папки Dataset_900/train и
     Dataset_900/validation (то самое разбиение 70/20/10 из документации).

  5. AMP (fp16) ВЫКЛЮЧЕН по умолчанию: это единственное численное отличие
     Docker-образа от оригинального репозитория, и именно оно усиливает
     нестабильность метрики. Включается флагом --amp.

  6. Детерминированная валидация: FPS со случайной стартовой точкой
     (torch.randint) заменён на детерминированный в eval-режиме, а
     подвыборка точек на validation фиксирована.
"""
import os
import csv
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

from dataloader.DataLoader_HybridTransformer import ObjDataset, patient_key
from networks.HybridTransformer import HybridPointTransformer


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class ConfusionMeter:
    """Метрики по матрице ошибок (confusion), накопленной за эпоху.

    mIoU считается один раз на весь набор: TP[c] / (TP[c] + FP[c] + FN[c]).
    Это НЕ то же самое, что прежнее "среднее по образцам от среднего по
    присутствующим классам", но оценка в сотни раз стабильнее.
    """

    def __init__(self, num_classes, device='cpu'):
        self.n = num_classes
        self.device = device
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    def update(self, pred, target):
        pred = pred.detach().view(-1)
        target = target.detach().view(-1)
        k = (target >= 0) & (target < self.n)
        pred, target = pred[k], target[k]
        idx = (target * self.n + pred).to(torch.long)
        self.conf += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self):
        conf = self.conf.double()
        tp = conf.diag()
        gt = conf.sum(1)          # истинных точек каждого класса
        pp = conf.sum(0)          # предсказанных точек каждого класса
        union = gt + pp - tp
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.zeros_like(tp))
        present = gt > 0
        miou = iou[present].mean().item() if present.any() else 0.0
        acc = (tp.sum() / conf.sum().clamp(min=1)).item()
        return acc, miou


class DentalSegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.best_miou = 0.0        # лучший "сырой" mIoU (для отчёта)
        self.best_score = 0.0       # лучший EMA-score (по нему early stopping)
        self.val_ema = None
        os.makedirs(config['save_path'], exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stop_counter = 0
        self.patience = config.get('early_stopping_patience', 20)
        self.ema_alpha = config.get('ema_alpha', 0.3)
        # AMP scaler для смешанной точности (включается только при наличии CUDA).
        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        # Диагностика AMP: считаем "переполнения" (inf/nan в градиентах), при
        # которых GradScaler ПРОПУСКАЕТ шаг оптимизатора. Если счётчик растёт
        # на каждой "плохой" эпохе - виноват fp16.
        self.amp_overflows = 0
        self._last_scale = self.scaler.get_scale() if self.use_amp else None

    # ------------------------------------------------------------------ metrics
    def compute_metrics(self, pred, target):
        """Старая (legacy) метрика: среднее по образцам. Оставлена для сравнения."""
        pred_classes = pred.argmax(dim=-1).view(-1)
        target_flat = target.view(-1)
        accuracy = (pred_classes == target_flat).float().mean().item()
        iou_per_class = []
        for cls in range(self.config['num_classes']):
            pred_inds = (pred_classes == cls)
            target_inds = (target_flat == cls)
            if target_inds.sum() > 0:
                intersection = (pred_inds[target_inds]).long().sum().item()
                union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
                iou_per_class.append(intersection / (union + 1e-8))
        return accuracy, np.mean(iou_per_class) if iou_per_class else 0.0

    def run_epoch(self, model, loader, criterion, optimizer, is_train, accumulation_steps=1):
        model.train(is_train)
        total_loss, total_acc, total_iou = 0.0, 0.0, 0.0
        desc = "Training" if is_train else "Validation"
        meter = ConfusionMeter(self.config['num_classes'], device=self.device)
        use_conf = self.config.get('metric', 'confusion') == 'confusion'

        if is_train:
            optimizer.zero_grad()

        for i, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            pos = batch['pos'].to(self.device)
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            with torch.set_grad_enabled(is_train), autocast(enabled=self.use_amp):
                outputs = model(pos, x)
                loss = criterion(outputs.reshape(-1, self.config['num_classes']), y.reshape(-1))

            if is_train:
                loss_scaled = loss / accumulation_steps
                self.scaler.scale(loss_scaled).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    if self.use_amp:
                        scale_now = self.scaler.get_scale()
                        if scale_now < self._last_scale:
                            # scale уменьшился => был inf/nan, шаг оптимизатора пропущен
                            self.amp_overflows += 1
                        self._last_scale = scale_now
                    optimizer.zero_grad()

            if use_conf:
                meter.update(outputs.argmax(dim=-1), y)
                acc = iou = 0.0
            else:
                acc, iou = self.compute_metrics(outputs, y)
            total_loss += loss.item() * accumulation_steps
            total_acc += acc
            total_iou += iou

        if use_conf:
            acc, miou = meter.compute()
            return {'loss': total_loss / len(loader), 'accuracy': acc, 'mIoU': miou}
        return {'loss': total_loss / len(loader), 'accuracy': total_acc / len(loader),
                'mIoU': total_iou / len(loader)}

    # ------------------------------------------------------------------ checkpoints
    def _state(self, epoch):
        return {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_miou': self.best_miou,
            'best_score': self.best_score,
            'early_stop_counter': self.early_stop_counter,
            'val_ema': float(self.val_ema) if self.val_ema is not None else 0.0,
            # Тип метрики: без него нельзя понять, сравнима ли best_mIoU
            # из старого чекпоинта с текущей метрикой.
            'metric': self.config.get('metric', 'confusion'),
        }

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return 0

        print(f"🔁 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully")

        # --- сравнимость порога early stopping ------------------------------
        # Если чекпоинт сохранён по ДРУГОЙ метрике (legacy mIoU по образцам vs
        # confusion-mIoU за эпоху), его best_mIoU нельзя сравнивать с текущей
        # метрикой: значения на разных шкалах. Иначе early stopping срабатывает
        # сразу или не срабатывает никогда.
        ckpt_metric = checkpoint.get('metric')
        cur_metric = self.config.get('metric', 'confusion')
        # У чекпоинтов, сохранённых ДО патча, ключа 'metric' нет - они всегда
        # считались legacy-метрикой (среднее по образцам).
        if ckpt_metric is None:
            ckpt_metric = 'legacy'
        metric_changed = ckpt_metric != cur_metric
        if metric_changed:
            print(f"⚠️  В чекпоинте метрика '{ckpt_metric}', а сейчас '{cur_metric}' — "
                  f"значения несопоставимы.")
            print("    Сбрасываю порог early stopping (best/patience/EMA) в 0.")
            print("    (Чтобы начать с чистого листа - добавьте --fresh_start.)")
        if 'best_miou' in checkpoint and not metric_changed:
            self.best_miou = checkpoint['best_miou']
            print(f"✅ Loaded best mIoU: {self.best_miou:.4f}")
        self.best_score = 0.0 if metric_changed else checkpoint.get('best_score', self.best_miou)
        if 'early_stop_counter' in checkpoint and not metric_changed:
            self.early_stop_counter = checkpoint['early_stop_counter']
            print(f"✅ Restored early-stop counter: {self.early_stop_counter}")
        if checkpoint.get('val_ema') and not metric_changed:
            self.val_ema = float(checkpoint['val_ema'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded")

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✅ AMP scaler state loaded")

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resuming from epoch {start_epoch}")
            return start_epoch

        print("⚠️ No epoch info in checkpoint. Starting from epoch 1.")
        return 1

    def _initialize_components(self, dataset, load_checkpoint=True):
        self.model = HybridPointTransformer(
            num_classes=self.config['num_classes'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            debug=self.config.get('debug_model', False)
        ).to(self.device)

        class_weights = dataset.get_label_weights().to(self.device)
        self.criterion = FocalLoss(gamma=self.config['focal_gamma'], alpha=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

        if not load_checkpoint:
            return 1

        save_path = self.config['save_path']
        last = os.path.join(save_path, 'checkpoint_last.pth')
        best = os.path.join(save_path, 'checkpoint_best.pth')

        mode = self.config.get('resume_mode', 'auto')
        if mode == 'none':
            print("▶️  --fresh_start: игнорирую существующие чекпоинты, обучение с нуля.")
            return 1
        if mode == 'weights_only':
            # Тёплый старт: берём ТОЛЬКО веса, сбрасываем всю обучающую
            # историю (best/patience/EMA/номер эпохи) - метрика и разбиение
            # изменились, старая история к ним неприменима.
            for p in (best, last):
                if os.path.exists(p):
                    print(f"▶️  --resume_weights: беру только веса из {os.path.basename(p)}")
                    return self._load_weights_only(p)
            print("⚠️  --resume_weights: чекпоинт не найден, обучение с нуля.")
            return 1

        # auto: сначала "последний" чекпоинт (реальное продолжение обучения),
        # и только если его нет - лучший (как раньше).
        checkpoint_path = last if os.path.exists(last) else best
        if not os.path.exists(checkpoint_path):
            print("▶️  Чекпоинт не найден - обучение с нуля.")
            return 1

        # Страховка: перед тем как новый код перезапишет checkpoint_best.pth,
        # сохраняем старую копию - чтобы веса до патча остались доступны
        # (например, чтобы сделать тёплый старт с другой метрикой).
        if checkpoint_path == best:
            backup = os.path.join(save_path, 'checkpoint_best_prepatch.pth')
            if os.path.exists(checkpoint_path) and not os.path.exists(backup):
                shutil.copy2(checkpoint_path, backup)
                print(f"🗄️  Старый checkpoint_best сохранён как {os.path.basename(backup)}")

        return self.load_checkpoint(checkpoint_path)

    def _load_weights_only(self, path):
        """Загрузить только веса модели, сбросив обучающее состояние."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_miou = 0.0
        self.best_score = 0.0
        self.early_stop_counter = 0
        self.val_ema = None
        print("✅ Веса загружены, обучающее состояние сброшено (best/patience/EMA = 0).")
        return 1

    # ------------------------------------------------------------------ split
    def _build_datasets(self):
        cfg = self.config
        train_augment = not cfg.get('no_augment', False)

        train_dataset = ObjDataset(
            obj_dir=cfg['data_path'],
            json_dir=cfg['gt_path'],
            num_points=cfg['num_points'],
            augment=train_augment,
            binary=cfg['binary'],
            num_classes=cfg['num_classes'],
            deterministic=False,
        )

        # --- валидация -------------------------------------------------------
        val_data_path = cfg.get('val_data_path')
        if val_data_path:
            if not cfg.get('val_gt_path'):
                raise SystemExit("❌ Указан --val_data_path, но не указан --val_gt_path")
            # Готовая папка из Split_Data_900.py: разбиение по пациентам.
            val_dataset = ObjDataset(
                obj_dir=val_data_path,
                json_dir=cfg.get('val_gt_path'),
                num_points=cfg['num_points'],
                augment=False,
                binary=cfg['binary'],
                num_classes=cfg['num_classes'],
                deterministic=True,
                seed=cfg.get('seed', 0),
                compute_weights=False,   # веса считаем один раз, на train
            )
            # На всякий случай кладём в val-датасет те же веса, что и в train.
            val_dataset.label_weights = train_dataset.label_weights
            val_indices = list(range(len(val_dataset)))
            train_indices = list(range(len(train_dataset)))
            print(f"✅ Using provided validation folder: {val_data_path} "
                  f"({len(val_dataset)} scans) - split is patient-level")
        else:
            # Ресplit внутри папки train - но ПО ПАЦИЕНТАМ, а не по файлам.
            val_dataset = ObjDataset(
                obj_dir=cfg['data_path'],
                json_dir=cfg['gt_path'],
                num_points=cfg['num_points'],
                augment=False,
                binary=cfg['binary'],
                num_classes=cfg['num_classes'],
                deterministic=True,
                seed=cfg.get('seed', 0),
                compute_weights=False,   # веса считаем один раз, на train
            )
            # На всякий случай кладём в val-датасет те же веса, что и в train.
            val_dataset.label_weights = train_dataset.label_weights
            train_indices, val_indices = self._patient_level_split(
                train_dataset, cfg['val_split'], cfg['test_split'], cfg.get('seed', 0))

        return train_dataset, val_dataset, train_indices, val_indices

    @staticmethod
    def _patient_level_split(dataset, val_split, test_split, seed=0):
        """Разбиение по пациентам: обе челюсти одного case идут в одну выборку."""
        patients = sorted({patient_key(p) for p in dataset.obj_files})
        rng = np.random.RandomState(seed)
        order = np.arange(len(patients))
        rng.shuffle(order)
        n_val = int(val_split * len(patients))
        n_test = int(test_split * len(patients))
        val_patients = {patients[i] for i in order[:n_val]}
        train_patients = {patients[i] for i in order[n_val + n_test:]}

        train_idx = [i for i, p in enumerate(dataset.obj_files) if patient_key(p) in train_patients]
        val_idx = [i for i, p in enumerate(dataset.obj_files) if patient_key(p) in val_patients]
        print(f"✅ Patient-level split: {len(train_patients)} train / {len(val_patients)} val patients "
              f"-> {len(train_idx)} train / {len(val_idx)} val scans")
        return train_idx, val_idx

    # ------------------------------------------------------------------ main loop
    def train(self):
        train_dataset, val_dataset, train_indices, val_indices = self._build_datasets()

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], sampler=train_sampler,
                                  num_workers=self.config['num_workers'], pin_memory=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], sampler=val_sampler,
                                num_workers=self.config['num_workers'], pin_memory=False, drop_last=True)

        log_file = os.path.join(self.config['save_path'], 'training_log.csv')
        header = ['Epoch', 'Train_Loss', 'Train_Acc', 'Train_mIoU',
                  'Val_Loss', 'Val_Acc', 'Val_mIoU', 'Val_mIoU_EMA']
        if os.path.exists(log_file):
            # Старый лог (7 колонок) несовместим с новым форматом -> сохраняем
            # его отдельно и начинаем новый, чтобы не портить историю.
            with open(log_file, 'r', newline='') as f:
                first = f.readline().strip()
            if first != ','.join(header):
                backup = os.path.join(self.config['save_path'], 'training_log_legacy.csv')
                i = 1
                while os.path.exists(backup):
                    backup = os.path.join(self.config['save_path'], f'training_log_legacy_{i}.csv')
                    i += 1
                os.replace(log_file, backup)
                print(f"⚠️ Старый формат лога ({first}) -> сохранён как {backup}, создан новый лог")
            else:
                print(f"📊 Continuing existing log file: {log_file}")
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(header)

        start_epoch = self._initialize_components(train_dataset, load_checkpoint=True)

        print("--- CONTINUING Training ---")
        print(f"Using device: {self.device}")
        print(f"Total steps per epoch: {len(train_loader)} | Training epochs: {start_epoch} to {self.config['epochs']}")
        print(f"Total additional epochs: {self.config['epochs'] - start_epoch + 1}")
        print(f"Metric: {self.config.get('metric', 'confusion')} | AMP: {self.use_amp}")

        print("\n🔍 Quick validation to verify model state...")
        self.model.eval()
        with torch.no_grad(), autocast(enabled=self.use_amp):
            sample_batch = next(iter(val_loader))
            pos = sample_batch['pos'].to(self.device)
            x = sample_batch['x'].to(self.device)
            sample_output = self.model(pos, x)
            sample_pred = sample_output.argmax(dim=-1)
            sample_acc = (sample_pred == sample_batch['y'].to(self.device)).float().mean().item()
            print(f"Sample validation accuracy: {sample_acc:.4f}")

        for epoch in range(start_epoch, self.config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")

            train_metrics = self.run_epoch(self.model, train_loader, self.criterion, self.optimizer,
                                           is_train=True,
                                           accumulation_steps=self.config['accumulation_steps'])
            print(f"Train | Loss: {train_metrics['loss']:.6f} | Acc: {train_metrics['accuracy']:.4f} "
                  f"| mIoU: {train_metrics['mIoU']:.4f}")
            if self.use_amp:
                print(f"      AMP | scale={self.scaler.get_scale():.0f} "
                      f"| skipped_steps(overflow)={self.amp_overflows}")

            val_metrics = self.run_epoch(self.model, val_loader, self.criterion, None, is_train=False,
                                         accumulation_steps=1)
            print(f"Val   | Loss: {val_metrics['loss']:.6f} | Acc: {val_metrics['accuracy']:.4f} "
                  f"| mIoU: {val_metrics['mIoU']:.4f}")

            # Сглаживание: одиночная "шумная" эпоха больше не решает судьбу
            # обучения, а плато считается по устойчивому тренду.
            raw = val_metrics['mIoU']
            self.val_ema = raw if self.val_ema is None else \
                self.ema_alpha * raw + (1 - self.ema_alpha) * self.val_ema
            score = self.val_ema

            is_new_best = score > self.best_score
            if is_new_best:
                self.best_score = score
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            self.best_miou = max(self.best_miou, raw)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    round(train_metrics['loss'], 6),
                    round(train_metrics['accuracy'], 6),
                    round(train_metrics['mIoU'], 6),
                    round(val_metrics['loss'], 6),
                    round(val_metrics['accuracy'], 6),
                    round(val_metrics['mIoU'], 6),
                    round(self.val_ema, 6),
                ])

            if is_new_best:
                print(f"🚀 New best model (EMA mIoU {self.best_score:.4f}, raw {raw:.4f}). "
                      f"Saving to {self.config['save_path']}")
                torch.save(self._state(epoch), os.path.join(self.config['save_path'], 'checkpoint_best.pth'))

            # Чекпоинт последней эпохи - для честного продолжения обучения.
            torch.save(self._state(epoch), os.path.join(self.config['save_path'], 'checkpoint_last.pth'))

            if (epoch + 1) % 5 == 0:
                backup_path = os.path.join(self.config['save_path'], f'backup_epoch_{epoch + 1}.pth')
                torch.save(self._state(epoch), backup_path)
                print(f"💾 Backup saved: {backup_path}")

            if self.early_stop_counter >= self.patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs "
                      f"(EMA mIoU не улучшался {self.patience} эпох подряд)")
                break

        print(f"\n--- TRAINING COMPLETED ---")
        print(f"Final best validation mIoU (raw): {self.best_miou:.4f}")
        print(f"Final best validation mIoU (EMA): {self.best_score:.4f}")
        print(f"Training completed from epoch {start_epoch} to {self.config['epochs']}")


def get_args():
    """CLI-аргументы. Все пути по умолчанию рассчитаны на монтирование датасета в /app/Dataset_900."""
    parser = argparse.ArgumentParser(
        description='Training HybridPointTransformer for Dental Segmentation (53 FDI classes)')
    # Пути к данным - датасет монтируется отдельно (например, через Hugging Face datasets).
    parser.add_argument('--data_path', type=str,
                        default=os.environ.get('DATA_PATH', './Dataset_900/train/data'),
                        help='Директория с .obj файлами обучающей выборки')
    parser.add_argument('--gt_path', type=str,
                        default=os.environ.get('GT_PATH', './Dataset_900/train/ground-truth'),
                        help='Директория с .json файлами разметки (ground truth)')
    parser.add_argument('--val_data_path', type=str,
                        default=os.environ.get('VAL_DATA_PATH', './Dataset_900/validation/data'),
                        help='Директория с .obj файлами валидации. Пусто -> ресplit внутри data_path')
    parser.add_argument('--val_gt_path', type=str,
                        default=os.environ.get('VAL_GT_PATH', './Dataset_900/validation/ground-truth'),
                        help='Директория с .json файлами валидации')
    parser.add_argument('--save_path', type=str,
                        default=os.environ.get('SAVE_PATH',
                                               './checkpoints/HybridTransformer_Final_6D_600_try5_53/'),
                        help='Куда сохранять чекпоинты и training_log.csv')
    # Гиперпараметры обучения
    parser.add_argument('--epochs', type=int, default=250, help='Кол-во эпох')
    parser.add_argument('--batch_size', type=int, default=1, help='Размер батча (Table 2: 1)')
    parser.add_argument('--num_workers', type=int, default=2, help='worker-ов у DataLoader')
    parser.add_argument('--num_points', type=int, default=65536, help='Точек на образец после сэмплинга')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (AdamW)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (AdamW)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Шагов накопления градиента')
    parser.add_argument('--focal_gamma', type=float, default=2.8214, help='Гамма FocalLoss')
    parser.add_argument('--d_model', type=int, default=256, help='Размерность модели трансформера')
    parser.add_argument('--nhead', type=int, default=4, help='Кол-во голов внимания')
    parser.add_argument('--val_split', type=float, default=0.2, help='Доля валидации (только при респите)')
    parser.add_argument('--test_split', type=float, default=0.1, help='Доля теста (только при респите)')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Ранний стоп после N эпох без улучшения')
    parser.add_argument('--ema_alpha', type=float, default=0.3, help='Коэффициент сглаживания val-метрики (EMA)')
    parser.add_argument('--seed', type=int, default=42, help='Seed для разбиения и детерминированной валидации')
    parser.add_argument('--metric', type=str, default='confusion', choices=['confusion', 'legacy'],
                        help='confusion - устойчивая mIoU за эпоху (по умолчанию); legacy - как раньше')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Видимые GPU (CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--resume_mode', type=str, default='auto',
                        choices=['auto', 'none', 'weights_only'],
                        help='auto - продолжить из checkpoint_last/best; none - с нуля (--fresh_start); '
                             'weights_only - взять только веса, сбросив early stopping')
    parser.add_argument('--fresh_start', action='store_true',
                        help='Игнорировать существующие чекпоинты и обучаться с нуля')
    parser.add_argument('--resume_weights', action='store_true',
                        help='Взять только веса из чекпоинта, сбросив early stopping и счётчики')
    parser.add_argument('--amp', action='store_true',
                        help='Включить смешанную точность (fp16). По умолчанию ВЫКЛЮЧЕНА')
    parser.add_argument('--no_amp', action='store_true',
                        help='Устарело: AMP и так выключена по умолчанию')
    parser.add_argument('--no_augment', action='store_true',
                        help='Отключить аугментации (включая ElasticDistortion). Полезно при отладке.')
    parser.add_argument('--debug_model', action='store_true', help='Отладочный режим модели')
    return parser.parse_args()


def main():
    args = get_args()

    # Ограничение видимых GPU.
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    resume_mode = 'auto'
    if args.fresh_start:
        resume_mode = 'none'
    elif args.resume_weights:
        resume_mode = 'weights_only'

    use_amp = bool(args.amp) and torch.cuda.is_available()
    if args.amp and not torch.cuda.is_available():
        print("⚠️ --amp requested but CUDA is not available -> AMP disabled")

    config = {
        'data_path': args.data_path,
        'gt_path': args.gt_path,
        'val_data_path': args.val_data_path if os.path.isdir(args.val_data_path or '') else None,
        'val_gt_path': args.val_gt_path,
        'save_path': args.save_path,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_amp': use_amp,
        'num_classes': 53,
        'binary': False,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'epochs': args.epochs,
        'accumulation_steps': args.accumulation_steps,

        'num_workers': args.num_workers,
        'num_points': args.num_points,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'focal_gamma': args.focal_gamma,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'early_stopping_patience': args.early_stopping_patience,
        'ema_alpha': args.ema_alpha,
        'seed': args.seed,
        'metric': args.metric,
        'resume_mode': resume_mode,
        'debug_model': args.debug_model,
        'no_augment': args.no_augment,
    }

    print("--- CONTINUING Training from Last Checkpoint (53 classes) ---")
    print(f"Using device: {config['device']}")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Data path:   {config['data_path']}")
    print(f"GT path:     {config['gt_path']}")
    print(f"Val path:    {config['val_data_path'] or '(resplit inside data_path, patient-level)'}")
    print(f"Save path:   {config['save_path']}")
    print(f"AMP enabled: {config['use_amp']}")
    print(f"Metric:      {config['metric']}")
    print(f"Resume mode: {config['resume_mode']}")

    trainer = DentalSegmentationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
