# HybridPointTransformer: Multi-class Segmentation on Digitalized 3D Dental Models

Official implementation of the paper **"HybridPointTransformer: Multi-class Segmentation on Digitalized 3D Dental Models"**.

This repository contains the training and evaluation code for a hybrid point-cloud architecture that combines PointNet++ style local feature aggregation with Transformer-based global attention for 53-class FDI-compliant dental segmentation (32 permanent teeth, 20 primary teeth, and the gingiva/background).

The model processes 65,536-point 6-D point clouds (XYZ + RGB) and is trained with a class-weighted focal loss tuned via Optuna. Reported results on the 3DTeethSeg'22 dataset (1,800 intraoral scans from 900 patients, 70/20/10 split): **92.71% overall accuracy and 89.51% mIoU**, outperforming PointNet, PointNet++, DGCNN, GNN, and KPConv baselines.

## Paper

- Title: *HybridPointTransformer: Multi-class Segmentation on Digitalized 3D Dental Models*
- Method: PointNet++ set abstraction encoder (1024 -> 256 -> 64 points) + 3 Transformer blocks (d_model=256, 4 heads) + PointNet feature-propagation decoder with skip connections
- Loss: class-weighted focal loss, `w_c = (1 / log(1.02 + n_c))` normalized, `gamma = 2.8214`
- Optimizer: AdamW (`lr = 1e-4`, `weight_decay = 1e-4`) or per-model Table 2 hyperparameters found by Optuna (50 trials, TPE)

## Results

| Model | Loss | Optimizer | LR | Weight Decay | Batch | Acc. | mIoU |
|---|---|---|---|---|---|---|---|
| HybridPointTransformer (Ours) | Focal | AdamW | 1e-4 | 1e-4 | 1 | 0.9271 | 0.8951 |
| PointNet | Focal | AdamW | 2.4e-5 | 2.1e-5 | 1 | 0.70 | 0.45 |
| PointNet++ | Focal | AdamW | 5.38e-4 | 1e-6 | 1 | 0.80 | 0.58 |
| DGCNN | Focal | AdamW | 1.1e-5 | 4.2e-5 | 1 | 0.71 | 0.49 |
| GNN (TeethGNN) | Cross-Entropy | Adam | 9.6e-5 | 8e-6 | 1 | 0.78 | 0.52 |
| KPConv | Cross-Entropy | AdamW | 2.8e-5 | 7.49e-4 | 1 | 0.65 | 0.42 |

## Requirements

- Python >= 3.8
- PyTorch >= 1.13 (образ зафиксирован на PyTorch 2.3.1 + CUDA 12.1)
- torch-geometric + `torch-scatter`/`torch-sparse`/`torch-cluster`/`torch-spline-conv`
- `torch_points3d` — **не требуется**. Класс `ElasticDistortion` реализован в `utils/elastic_distortion.py`:
  точная копия алгоритма upstream (`granularity` — размер ячейки шума, `magnitude` — амплитуда смещения,
  обе величины в единицах координат), на чистом numpy, без scipy и без тяжёлой сборки сетки шума.
  Если настоящий `torch-points3d` установлен, его можно включить переменной окружения `TP3D_USE_UPSTREAM=1`
  (при неверной форме выхода модуль сам откатится на локальную реализацию).
- open3d, trimesh, scikit-learn (graph preprocessing)
- numpy, tqdm, pandas, einops
- optuna (hyperparameter optimization)

Полный список — в `requirements.txt`.

## Dataset

The public **3DTeethSeg'22** challenge dataset (MICCAI 2022) is used. Each sample consists of:

- `data/<case>/<case>_upper.obj` / `<case>_lower.obj` — 6-D vertex clouds (XYZ + RGB)
- `ground-truth/<case>/<case>_upper.json` / `<case>_lower.json` — per-vertex FDI labels

Prepare the 70/20/10 train/validation/test split at the patient level:

```
python Split_Data_900.py
```

This creates `Dataset_900/{train,validation,test}/{data,ground-truth}` plus per-scan mapping lists `train.txt`, `validation.txt`, `test.txt`. The final test scripts evaluate on `Dataset_900/test`.

## Repository Structure

```
dataloader/
  PreprocessingPaper.py          # shared Algorithm-1 preprocessing for all point-based models
  DataLoader_PNet.py             # PointNet
  DataLoader_PP.py               # PointNet++
  DataLoader_Cust_DGCNN.py       # DGCNN
  DataLoader_kpconv.py           # KPConv
  DataLoader_HybridTransformer.py# HybridPointTransformer (reference pipeline)
  Dataloader_TeethGnn.py         # raw facet-graph construction (10k facets)
  Dataloader_TeethGnn_Cached.py  # cached .pt graph dataset
networks/
  HybridTransformer.py           # proposed model
  PointNet.py, PointNetPlusPlus.py, Cust_DGCNN_Dilated.py, kpconv.py,
  TeethGnn_Model.py, TransformerNet.py, SuperpointTransformer.py, DentalMAN.py
Train_*.py                       # training + Optuna search for each model
Test_Hybrid.py, Test_PointNet.py, Test_PointNetPP.py, Test_Cust_DGCNN.py,
Test_kpconv.py, Test_TeethGnn.py # evaluation on the held-out test split
TeethGnnPreprocess_dataset.py    # build the TeethGNN .pt graph cache
Split_Data_900.py                # patient-level 70/20/10 split
```

## Preprocessing (Algorithm 1)

All point-based models share `PreprocessingPaper.py`:

1. Read 6-D vertex features (XYZ + RGB) and ground-truth labels.
2. Training only: random Z-rotation within +/-10 deg, scaling in [0.9, 1.1], color jitter, and elastic distortion.
3. Centre on the centroid and normalise to unit radius.
4. Remap FDI codes to [0, 52] (53 classes).
5. Sample each scan to 65,536 points (uniform sampling after a 100k-vertex cap).

Class weights `w_c = (1 / log(1.02 + n_c)) / sum_c(...) * 53` are computed from the label distribution (Eq. 2 of the paper).

See **[Experimental_Setup.md](Experimental_Setup.md)** for the complete, reproducible experimental setup with all hyperparameters, data paths, preprocessing details, and a full list of conflicts between the paper and the code.

## Training

Run each model with its paper-reported hyperparameters (set `--optuna_trials 0`) or launch the Optuna search (default 50 trials):

```bash
# Ours (HybridPointTransformer) — Table 2 config hard-coded in Train_Hybrid.py
python Train_Hybrid.py

# PointNet (Focal / AdamW / 2.4e-5 / 2.1e-5 / batch 1)
python Train_PointNet.py --optuna_trials 0 --epochs 150

# PointNet++ (Focal / AdamW / 5.38e-4 / 1e-6)
python Train_PointNetPP.py --optuna_trials 0 --epochs 150

# DGCNN (Focal / AdamW / 1.1e-5 / 4.2e-5)
python Train_Cust_DGCNN.py --optuna_trials 0 --epochs 150

# GNN / TeethGNN (Cross-Entropy / Adam / 9.6e-5 / 8e-6) — requires the graph cache first
python TeethGnnPreprocess_dataset.py
python Train_TeethGnn.py --optuna_trials 0 --epochs 150

# KPConv (Cross-Entropy / AdamW / 2.8e-5 / 7.49e-4)
python Train_kpconv.py --optuna_trials 0 --epochs 150
```

All scripts accept `--save_path`, `--experiment_name`, `--gpu_ids`, and `--epochs`. The Optuna search space follows the paper: learning rate `1e-6 .. 1e-2` (log), batch size `1`, optimizer `{Adam, AdamW}`, weight decay `1e-6 .. 1e-3` (log), focal gamma `1.0 .. 3.0`.

## Testing

The test scripts evaluate on the 10% held-out test split (`Dataset_900/test`) with `augment=False`, 65,536 points, and report overall accuracy and per-class mean IoU into `test_results/`:

```bash
python Test_Hybrid.py
python Test_PointNet.py
python Test_PointNetPP.py
python Test_Cust_DGCNN.py
python Test_kpconv.py
python Test_TeethGnn.py
```

`Test_TeethGnn.py` automatically builds the test graph cache on first run.

## Docker

В репозитории есть `Dockerfile`, `requirements.txt`, `.dockerignore` и GitHub Actions workflow `.github/workflows/docker-build.yml` (сборка и пуш в `ghcr.io/<owner>/<repo>:latest` + `:sha-<commit>` при пуше в `main`).

### Сборка

```bash
docker build -t hybrid-teeth:latest .
```

Базовый образ — `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` (PyTorch 2.3.1 + CUDA 12.1). PyG-стек (`torch-scatter`/`torch-sparse`/`torch-cluster`/`torch-spline-conv`/`torch-geometric`) ставится отдельным `RUN` с правильным индексом колёс `torch-2.3.1+cu121`. Ожидаемый размер образа — 10–12 GB.

### Точки монтирования

`Dockerfile` объявляет три `VOLUME`:

| Путь в контейнере | Назначение |
|---|---|
| `/app/Dataset_900` | Датасет (монтируется read-only; в образ не входит — исключён в `.dockerignore`) |
| `/app/checkpoints` | Чекпоинты и `training_log.csv` (иначе пропадут после остановки контейнера) |
| `/app/experiments` | Артефакты Optuna-экспериментов |

Датасет монтируется отдельно — например, через Hugging Face datasets, скачанный и распакованный на хост, либо напрямую volume.

### Точка входа (ENTRYPOINT)

```dockerfile
ENTRYPOINT ["python"]
CMD ["Train_Hybrid.py"]
```

Поэтому можно запускать **любой** скрипт репозитория, передав его как аргумент `docker run`:

```bash
# Обучение по умолчанию (Train_Hybrid.py):
docker run --gpus all --rm --shm-size=16g \
  -v /host/Dataset_900:/app/Dataset_900:ro \
  -v /host/checkpoints:/app/checkpoints \
  hybrid-teeth:latest

# Запуск другого скрипта — просто передаём его имя:
docker run --gpus all --rm --shm-size=16g \
  -v /host/Dataset_900:/app/Dataset_900:ro \
  -v /host/checkpoints:/app/checkpoints \
  hybrid-teeth:latest Test_Hybrid.py

# Обучение с переопределёнными параметрами:
docker run --gpus all --rm --shm-size=16g \
  -v /host/Dataset_900:/app/Dataset_900:ro \
  -v /host/checkpoints:/app/checkpoints \
  hybrid-teeth:latest Train_Hybrid.py --epochs 50 --save_path /app/checkpoints/exp1
```

- `--gpus all` — проброс GPU (требуется `nvidia-container-toolkit`).
- `--shm-size=16g` — PyTorch DataLoader с `num_workers>0` использует `/dev/shm`; дефолта 64 MB не хватит.

### CLI-аргументы `Train_Hybrid.py`

`Train_Hybrid.py` теперь принимает CLI-аргументы (все пути также читаются из env-переменных, если аргумент не передан):

| Аргумент | Env-переменная | По умолчанию |
|---|---|---|
| `--data_path` | `DATA_PATH` | `./Dataset_900/train/data` |
| `--gt_path` | `GT_PATH` | `./Dataset_900/train/ground-truth` |
| `--save_path` | `SAVE_PATH` | `./checkpoints/HybridTransformer_Final_6D_600_try5_53/` |
| `--epochs` | — | `250` |
| `--batch_size` | — | `1` |
| `--num_workers` | — | `2` |
| `--num_points` | — | `65536` |
| `--lr` | — | `1e-4` |
| `--weight_decay` | — | `1e-4` |
| `--accumulation_steps` | — | `1` |
| `--focal_gamma` | — | `2.8214` |
| `--d_model` | — | `256` |
| `--nhead` | — | `4` |
| `--val_split` | — | `0.2` |
| `--test_split` | — | `0.1` |
| `--early_stopping_patience` | — | `20` |
| `--ema_alpha` | — | `0.3` (коэффициент сглаживания val-метрики для early stopping) |
| `--seed` | — | `42` (разбиение по пациентам + детерминированная валидация) |
| `--metric` | — | `confusion` (устойчивая mIoU за эпоху) \| `legacy` (как в оригинале) |
| `--val_data_path` | `VAL_DATA_PATH` | `./Dataset_900/validation/data` |
| `--val_gt_path` | `VAL_GT_PATH` | `./Dataset_900/validation/ground-truth` |
| `--amp` | — | (флаг) **включить** смешанную точность (fp16); по умолчанию ВЫКЛЮЧЕНА |
| `--resume_mode` | — | `auto` (продолжить из `checkpoint_last`/`best`) \| `none` (с нуля) \| `weights_only` (только веса) |
| `--fresh_start` | — | (флаг) игнорировать существующие чекпоинты, обучаться с нуля |
| `--resume_weights` | — | (флаг) взять только веса из чекпоинта, сбросив early stopping |
| `--gpu_ids` | — | `0` |
| `--no_amp` | — | устарело: AMP и так выключена по умолчанию |
| `--no_augment` | — | (флаг) отключить аугментации (включая ElasticDistortion) |
| `--debug_model` | — | (флаг) отладочный режим модели |

Смешанная точность (AMP) **выключена по умолчанию**: в оригинальном репозитории ключ `use_amp` присутствовал в конфиге `Train_Hybrid.py`, но `autocast` там не использовался — то есть референсный прогон главной модели шёл в **fp32**. В 8 базовых скриптах (`Train_PointNetPP.py`, `Train_kpconv.py` и др.) AMP, наоборот, используется, так что fp16 сам по себе не «сломан» — но для `Train_Hybrid.py` это новое численное отличие от оригинала, которое стоит проверить отдельным прогоном. Включается флагом `--amp` (состояние `GradScaler` сохраняется и загружается в чекпоинтах).

> **Рекомендация:** сначала прогнать обучение **без** AMP и убедиться, что val-график ровный. Если рядом с `Val mIoU` печатается строка `AMP | scale=... | skipped_steps(overflow)=N` и счётчик растёт на «плохих» эпохах — fp16 подтверждён как виновник, и AMP лучше оставить выключенным.

### Аугментации и ElasticDistortion

Аугментации (`_augment_features` + `ElasticDistortion`) применяются **только к train-датасету**; для val-датасета `augment=False` (см. `Train_Hybrid.py`, два отдельных `ObjDataset`). Раньше единый датасет с `augment=True` кормил оба сэмплера, и первый же val-прогон каскадно падал ~252 раза.

`ElasticDistortion` живёт в `utils/elastic_distortion.py` (папка-шим `torch_points3d/` удалена — она перекрывала настоящий пакет). Реализация повторяет upstream-семантику: `granularity` — размер ячейки шума, `magnitude` — амплитуда смещения, **обе величины в единицах координат (мм)**; итоговое смещение `interp(coords) * magnitude`. Раньше в шиме было `disp * magnitude * extent_bbox` — на сканах 3DTeethSeg (размах 60–80 мм) это давало смещение до 100 мм и полностью разрушало геометрию зубов, из-за чего модель выучивала только априорный класс (фон): val accuracy ≈ доля фона, val mIoU ≈ prior baseline.

Модуль работает на чистом numpy (без scipy), сетка шума ограничена `MAX_GRID_CELLS=128`, выходная форма валидируется. Если установлен настоящий `torch-points3d`, его можно включить через `TP3D_USE_UPSTREAM=1` — тогда вызывается upstream, а при неверной форме `pos` происходит автоматический откат на локальную реализацию. Для отладки аугментации полностью отключаются флагом `--no_augment`.

### Пути в других `Train_*.py` / `Test_*.py`

Все 16 обучающих/тестовых скриптов теперь используют относительные пути по умолчанию (раньше 14 из них содержали Windows-пути `K:/...`). Пути к данным можно переопределить:

- Скрипты с `argparse` (`Train_Cust_DGCNN.py`, `Train_PointNet.py`, `Train_PointNetPP.py`, `Train_kpconv.py`, `Train_TeethGnn.py`, `Train_Hybrid.py`) — через `--data_path` / `--gt_path` или env-переменные `DATA_PATH` / `GT_PATH`.
- Скрипты без `argparse` (`Train_DentalMAN.py`, `Train_SPT.py`, `Train_Transformer.py`, `Test_DentalMAN.py`, `Test_HybridByIndividual.py`, `Test_HybridByIndividualSaveDirect.py`) — через env-переменные `DATA_PATH` / `GT_PATH`.
- Тестовые скрипты с `DATASET_ROOT` (`Test_Cust_DGCNN.py`, `Test_PointNet.py`, `Test_PointNetPP.py`, `Test_TeethGnn.py`, `Test_kpconv.py`) — через env-переменную `DATASET_ROOT` (по умолчанию `./Dataset_900`).

Пример запуска теста с переопределённым корнем датасета:

```bash
docker run --gpus all --rm --shm-size=16g \
  -e DATASET_ROOT=/app/Dataset_900 \
  -v /host/Dataset_900:/app/Dataset_900:ro \
  -v /host/checkpoints:/app/checkpoints \
  hybrid-teeth:latest Test_kpconv.py
```

### Версии по умолчанию `num_classes`

Во всех сетях (`HybridTransformer.py`, `SuperpointTransformer.py`, `TeethGnn_Model.py`, `TransformerNet.py`, `Cust_DGCNN_Dilated.py`, `DentalMAN.py`) значение `num_classes` по умолчанию приведено к **53** (раньше в четырёх сетях стояло `35`), чтобы инстанцирование без аргументов сразу давало корректную FDI-53 модель.

## Исправления от 26.11.2026 (см. `docs/DIAGNOSTIC.md`)

Полный разбор — в [`docs/DIAGNOSTIC.md`](docs/DIAGNOSTIC.md). Кратко, что было исправлено:

| Проблема | Симптом | Исправление |
|---|---|---|
| Незавершённый `git merge` | `utils/elastic_distortion.py` содержал маркеры конфликта → `SyntaxError` → обучение не стартует | Конфликт разрешён: векторизованный код + `MAX_GRID_CELLS=128` (ближе к upstream, чем 64) |
| Early stopping по шумной метрике | val mIoU скакал ±0.15 (sd 0.044) с периодом 4 эпохи → обучение останавливалось на 35-й эпохе, не выйдя на плато | Метрика теперь считается по матрице ошибок (confusion), накопленной за всю эпоху; early stopping — по EMA; `early_stop_counter` восстанавливается из чекпоинта |
| Ловушка resume | Продолжение шло из `checkpoint_best.pth` → каждый перезапуск откатывался на лучшую эпоху и снова упирался в early stopping | Resume теперь из `checkpoint_last.pth`; лучший чекпоинт хранится отдельно |
| Недетерминированная валидация | `furthest_point_sample` стартовал от `torch.randint` → набор центроидов менялся каждую эпоху | В eval-режиме FPS детерминирован; подвыборка 65 536 точек на validation фиксирована |
| Утечка в разбиении | Train/val резались **по файлам** → 75 % val-пациентов были и в train | Разбиение **по пациентам**; по умолчанию используется готовая папка `Dataset_900/validation` (patient-level, из `Split_Data_900.py`) |
| fp16 (AMP) | Только новое численное отличие от оригинала | AMP выключена по умолчанию (`--amp` включает) + диагностика переполнений |

Изменение определения метрики: `Val_mIoU` теперь это `IoU = TP/(TP+FP+FN)`, посчитанный по всей валидационной выборке, а не среднее по образцам. **Числовое значение другое** — с прежним результатом напрямую не сравнивайте. Прежний вариант доступен как `--metric legacy`. Для сопоставления с публикацией 3DTeethSeg'22 считайте метрики на `Dataset_900/test` через `Test_Hybrid.py`.

Формат `training_log.csv` расширен до 9 колонок (добавлена `Val_mIoU_EMA`). Старый лог автоматически сохраняется как `training_log_legacy.csv`.

---

## Как запустить после обновления (27.11.2026)

### Важно: старый чекпоинт нельзя просто «продолжить»

В бакете `dental-checkpoints` лежит `checkpoint_best.pth` от прогона, который:
- обучался на **разбиении по файлам** (75 % val-пациентов были и в train);
- измерялся **legacy-метрикой** (среднее по образцам), а не confusion-mIoU.

Значение `best_miou = 0.3527` из этого чекпоинта **несопоставимо** с новой метрикой. Если продолжить с него, порог early stopping будет сравниваться с числом на другой шкале. Поэтому:

- **рекомендуется начать с чистого листа** — старые веса обучены на залитой выборке, а разбиение и метрика теперь другие;
- если всё же нужен тёплый старт: `--resume_weights` (берутся только веса, вся обучающая история сбрасывается);
- защита от сравнения несопоставимых чисел встроена: при загрузке чекпоинта с другим типом метрики порог early stopping автоматически сбрасывается в 0 с предупреждением.

Самый чистый способ — **новый `--save_path`**, тогда старые артефакты в бакете останутся нетронутыми:

```bash
python Train_Hybrid.py --save_path /app/checkpoints/run2
```

### Пайплайн публикации

Образ `ghcr.io/bitnomichael/toothseparationdocker:latest` собирается GitHub Actions при пуше в ветку `main` (`.github/workflows/docker-build.yml`), теги: `latest` и `sha-<commit>`. Поэтому порядок такой:

1. запушить исправленный репозиторий в `main`;
2. дождаться сборки и пуша образа в GHCR;
3. запускать обучение, указав `:sha-<commit>` — так вы зафиксируете ровно тот образ, с которым работаете.

### HF Jobs: таймаут и возобновление

`--timeout 24h` обрывает джобу на ровно 24 часах. С новым `checkpoint_last.pth` джобу можно перезапускать: обучение продолжится с последней завершённой эпохи, а счётчик patience восстановится из чекпоинта (раньше он сбрасывался, и из-за этого каждый перезапуск откатывался на лучшую эпоху и заново упирался в early stopping — это и было причиной «обучение снова встало»).

Учтите, что после перехода на patient-level split в эпоху обрабатывается больше сканов (весь `Dataset_900/train` = 1260 вместо 882 + валидация 360), а `ElasticDistortion` стоит ~440 мс на скан. Если эпоха не укладывается — увеличьте `--num_workers` (2 → 4-8): аугментация однопоточная и упирается в CPU, а не в GPU.

### Контроль первых эпох

1. В стартовом логе должно быть `✅ Using provided validation folder: ... (360 scans) - split is patient-level`. Если папки `validation` в датасете нет — будет `✅ Patient-level split: ...`, это тоже корректно, но проверьте, что монтируется правильный датасет.
2. `Val mIoU` должен расти плавно, без скачков ±0.15.
3. `checkpoint_last.pth` появляется после первой эпохи — это то, что делает возобновление рабочим.

---

## Citation

If you use this code or the HybridPointTransformer in your research, please cite the paper (bibtex to be added).

## License

The code is released for research purposes. Please contact the authors for commercial use."# toothSeparationDocker" 
