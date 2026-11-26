#!/usr/bin/env python3
"""
Диагностика пайплайна сегментации зубов (3DTeethSeg'22, 53 класса).

Проверяет 4 гипотезы, которые объясняют "плато" mIoU ~0.031 при accuracy ~0.44:

  H1. Разметка не выровнена по вершинам: len(json['labels']) != число 'v' строк в .obj
  H2. Модель выродилась в предсказание одного класса (background) для всех точек
  H3. ElasticDistortion применяется в сырых единицах (мм) и разрушает геометрию
  H4. Классовые веса (Eq.2) почти не перебалансируют фон (44:1 -> 1.2:1)

Запуск:
  python diagnose_teeth_seg.py --data_path ./Dataset_900/train/data \
                               --gt_path   ./Dataset_900/train/ground-truth \
                               --max_files 120
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

FDI_TO_IDX = {}
FDI_TO_IDX[0] = 0
for _fdi in range(11, 19):
    FDI_TO_IDX[_fdi] = _fdi - 10          # 11-18 -> 1-8
for _fdi in range(21, 29):
    FDI_TO_IDX[_fdi] = _fdi - 20 + 8      # 21-28 -> 9-16
for _fdi in range(31, 39):
    FDI_TO_IDX[_fdi] = _fdi - 30 + 16     # 31-38 -> 17-24
for _fdi in range(41, 49):
    FDI_TO_IDX[_fdi] = _fdi - 40 + 24     # 41-48 -> 25-32
for _fdi in range(51, 56):
    FDI_TO_IDX[_fdi] = _fdi - 50 + 32     # 51-55 -> 33-37
for _fdi in range(61, 66):
    FDI_TO_IDX[_fdi] = _fdi - 60 + 37     # 61-65 -> 38-42
for _fdi in range(71, 76):
    FDI_TO_IDX[_fdi] = _fdi - 70 + 42     # 71-75 -> 43-47
for _fdi in range(81, 86):
    FDI_TO_IDX[_fdi] = _fdi - 80 + 47     # 81-85 -> 48-52
NUM_CLASSES = 53


def remap(labels):
    out = np.full(len(labels), -1, dtype=np.int64)
    for fdi, idx in FDI_TO_IDX.items():
        out[labels == fdi] = idx
    return out


# ============================ варп-функции (numpy only) ============================
def _tri_sample(grid, coords):
    """Трилинейная интерполяция: grid (nx,ny,nz), coords (3,N) в индексах сетки."""
    shape = np.array(grid.shape)
    c = [np.clip(coords[d], 0.0, shape[d] - 1.0 - 1e-6) for d in range(3)]
    i0 = [np.floor(c[d]).astype(np.int64) for d in range(3)]
    f = [c[d] - i0[d] for d in range(3)]
    out = np.zeros(coords.shape[1])
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                w = ((1 - f[0]) if bx == 0 else f[0]) * \
                    ((1 - f[1]) if by == 0 else f[1]) * \
                    ((1 - f[2]) if bz == 0 else f[2])
                out += w * grid[i0[0] + bx, i0[1] + by, i0[2] + bz]
    return out


def _smooth_axis(arr, axis):
    a = np.moveaxis(arr, axis, 0)
    z = np.zeros((1,) + a.shape[1:], dtype=a.dtype)
    pad = np.concatenate([z, a, z], axis=0)
    return np.moveaxis((pad[:-2] + 2 * pad[1:-1] + pad[2:]) * 0.25, 0, axis)


def warp_old(points, granularity=(0.2, 0.8), magnitude=(0.4, 1.6)):
    """СТАРЫЙ шим из репо: disp * magnitude * extent_bbox."""
    mn = points.min(0); mx = points.max(0)
    scale = np.maximum(mx - mn, 1e-8)
    norm = (points - mn) / scale
    gv = float(np.random.uniform(min(granularity), max(granularity)))
    g = int(np.ceil(1.0 / gv)) + 2
    mv = float(np.random.uniform(min(magnitude), max(magnitude)))
    rg = np.random.uniform(-1, 1, size=(g, g, g, 3)).astype(np.float32)
    coords = (norm * (np.array([g, g, g]) - 1)).T
    disp = np.zeros_like(norm)
    for d in range(3):
        disp[:, d] = _tri_sample(rg[..., d], coords)
    return points + disp * mv * scale


def warp_new(points, granularity=(0.2, 0.8), magnitude=(0.4, 1.6), max_cells=128):
    """НОВЫЙ (канон torch-points3d): disp * magnitude, где magnitude - амплитуда в мм."""
    mn = points.min(0)
    ext = (points - mn).max(0)
    gv = max(float(np.random.uniform(min(granularity), max(granularity))), 1e-6)
    mv = float(np.random.uniform(min(magnitude), max(magnitude)))
    eff = max(gv, float(ext.max()) / max_cells)
    nd = np.maximum((ext / eff).astype(np.int64) + 3, 3)
    noise = np.random.randn(*nd, 3).astype(np.float32)
    for ax in range(3):
        noise = _smooth_axis(noise, ax)
    ax_min = mn - eff
    ax_max = mn + eff * (nd - 2)
    coords = np.empty((points.shape[0], 3))
    for d in range(3):
        span = max(ax_max[d] - ax_min[d], 1e-8)
        coords[:, d] = (points[:, d] - ax_min[d]) * (nd[d] - 1) / span
    disp = np.empty_like(coords)
    for d in range(3):
        disp[:, d] = _tri_sample(noise[..., d], coords.T)
    return points + disp * mv


def read_obj_xyz(path):
    xyz = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':
                try:
                    vals = list(map(float, parts[1:7]))
                except ValueError:
                    continue
                xyz.append(vals)
    return np.asarray(xyz, dtype=np.float64)


def read_labels(path):
    with open(path, 'r') as f:
        return np.asarray(json.load(f)['labels'], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--gt_path', required=True)
    ap.add_argument('--max_files', type=int, default=120)
    ap.add_argument('--num_points', type=int, default=65536)
    args = ap.parse_args()

    objs = []
    for root, _, files in os.walk(args.data_path):
        for name in sorted(files):
            if name.endswith('.obj'):
                rel = os.path.relpath(os.path.join(root, name), args.data_path)
                gt = os.path.join(args.gt_path, os.path.splitext(rel)[0] + '.json')
                if os.path.exists(gt):
                    objs.append((os.path.join(root, name), gt))
    objs = objs[:args.max_files]
    if not objs:
        raise SystemExit('Не найдено ни одной пары .obj/.json')
    print(f'Сканирований к анализу: {len(objs)} (всего в каталоге см. ниже)\n')

    # ---------------- H1: выравнивание вершин и меток ----------------
    print('=' * 72)
    print('H1. Выравнивание разметки по вершинам')
    print('=' * 72)
    mismatched = []
    for obj_p, gt_p in objs:
        n_xyz = len(read_obj_xyz(obj_p))
        n_lab = len(read_labels(gt_p))
        if n_xyz != n_lab:
            mismatched.append((os.path.basename(obj_p), n_xyz, n_lab))
    if mismatched:
        print(f'!! РАСМИНОВАНИЕ в {len(mismatched)}/{len(objs)} файлов:')
        for name, n_xyz, n_lab in mismatched[:10]:
            print(f'   {name}: вершин в .obj = {n_xyz}, меток в .json = {n_lab} '
                  f'(разница {n_xyz - n_lab})')
        print('   -> лейблы НЕ соответствуют точкам: min_len-обрезка в DataLoader')
        print('      сдвигает метки, и задача становится неучимой (accuracy упирается в фон).')
    else:
        print('OK: во всех проверенных файлах число меток == число вершин.')

    # ---------------- H3: единицы и разрушающая аугментация ----------------
    print()
    print('=' * 72)
    print('H3. Масштаб координат и величина ElasticDistortion')
    print('=' * 72)
    extents, coords_all = [], []
    for obj_p, _ in objs[:min(len(objs), 20)]:
        xyz = read_obj_xyz(obj_p)
        if len(xyz) == 0:
            continue
        ptp = np.ptp(xyz[:, :3], axis=0)
        extents.append(ptp)
        coords_all.append(xyz[:, :3])
    if extents:
        e = np.asarray(extents)
        print(f'Размах координат .obj по осям: медиана = {np.median(e, axis=0).round(2)}, '
              f'max = {e.max(axis=0).round(2)}')
        med_ext = float(np.median(e))
        if med_ext > 5:
            print(f'-> координаты в МИЛЛИМЕТРАХ (размах ~{med_ext:.0f}).')
            print(f'   ElasticDistortion в текущем шиме: disp * magnitude * scale,')
            print(f'   magnitude in [0.4, 1.6] -> смещение до {1.6 * med_ext:.0f} мм,')
            print(f'   т.е. до {1.6 * med_ext / med_ext:.1f} размера bounding box.')
            print('   ГЕОМЕТРИЯ РАЗРУШАЕТСЯ.')
            print('   В upstream torch-points3d magnitude - это амплитуда в мм, а НЕ множитель')
            print('   размера bounding box: при magnitude 0.4-1.6 мм смещение ~0.2-0.5 мм (см. H3b).')
        else:
            print(f'-> координаты похожи на нормализованные (размах ~{med_ext:.2f}): '
                  f'ElasticDistortion корректен.')

    # ---------------- H3b: реальное смещение точек при варпе ----------------
    print()
    print('=' * 72)
    print('H3b. Фактическое смещение точек: старый шим vs канонический')
    print('=' * 72)
    for obj_p, _ in objs[:min(len(objs), 5)]:
        xyz = read_obj_xyz(obj_p)
        if len(xyz) < 100:
            continue
        xyz = xyz[:, :3].astype(np.float64)
        sub = xyz[np.random.choice(len(xyz), min(40000, len(xyz)), replace=False)]
        d_old = np.linalg.norm(warp_old(sub) - sub, axis=1)
        d_new = np.linalg.norm(warp_new(sub) - sub, axis=1)
        # характерный размер зуба ~ доля от extent
        step = float(np.median(np.ptp(xyz, axis=0))) / 10.0
        print(f'{os.path.basename(obj_p)} (шаг по дуге ~{step:.1f} мм)')
        print(f'   старый шим : mean={d_old.mean():7.2f} мм  p95={np.percentile(d_old,95):7.2f}  '
              f'max={d_old.max():7.2f}  | >2x-шаг: {100*(d_old>2*step).mean():5.1f}%')
        print(f'   новый  шим : mean={d_new.mean():7.2f} мм  p95={np.percentile(d_new,95):7.2f}  '
              f'max={d_new.max():7.2f}  | >2x-шаг: {100*(d_new>2*step).mean():5.1f}%')
    print('   !! Старый шим сдвигает каждую точку сильнее, чем размер зуба,')
    print('      и делает это гладким глобальным варпом -> геометрия дуги разрушена.')

    # ---------------- H2/H4: распределение классов и baseline ----------------
    print()
    print('=' * 72)
    print('H2/H4. Распределение классов и baseline "всегда самый частый класс"')
    print('=' * 72)
    total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_file_counts = []
    n_present_list, pred_hist = [], []
    for obj_p, gt_p in objs:
        labels = read_labels(gt_p)
        labels = remap(labels)
        if (labels < 0).any():
            print(f'   предупреждение: {os.path.basename(gt_p)} содержит '
                  f'{int((labels < 0).sum())} меток вне FDI-таблицы')
            labels = labels[labels >= 0]
        if len(labels) == 0:
            continue
        if len(labels) > args.num_points:
            idx = np.random.choice(len(labels), args.num_points, replace=False)
            labels = labels[idx]
        cnt = np.bincount(labels, minlength=NUM_CLASSES)
        total_counts += cnt
        per_file_counts.append(cnt)
        n_present_list.append(int((cnt > 0).sum()))
        pred_hist.append(int(cnt.argmax()))

    per_file_counts = np.asarray(per_file_counts)
    grand = total_counts.sum()
    order = np.argsort(-total_counts)
    print(f'Всего точек: {grand:,}')
    print(f'Доля фона (класс 0): {total_counts[0] / grand:.4f}')
    print(f'Классов в датасете (непустых): {int((total_counts > 0).sum())} / {NUM_CLASSES}')
    print(f'Среднее число присутствующих классов в скане: {np.mean(n_present_list):.2f} '
          f'(median {np.median(n_present_list):.0f})')
    print('\nТоп-8 классов по числу точек:')
    for c in order[:8]:
        share = total_counts[c] / grand
        print(f'   класс {c:2d} (FDI {c}): {total_counts[c]:>12,}  {share:7.4%}')
    print('\nРедкие классы (<0.01% точек):')
    rare = [c for c in range(NUM_CLASSES) if 0 < total_counts[c] / grand < 0.0001]
    print('   ' + (', '.join(str(c) for c in rare) if rare else 'нет'))

    # baseline: всегда предсказывать один класс (фон) для ВСЕХ точек
    base_acc, base_miou_present, base_miou_all = [], [], []
    for cnt in per_file_counts:
        n = cnt.sum()
        c = 0 if cnt[0] == cnt.max() else int(cnt.argmax())   # класс, покрывающий больше всего точек
        acc = cnt[c] / n
        base_acc.append(acc)
        # |pred_c| = n (мы предсказываем c для КАЖДОЙ точки), |target_p| = cnt[p]
        inter_c = cnt[c]
        ious_present, ious_all = [], []
        for p_ in range(NUM_CLASSES):
            inter = inter_c if p_ == c else 0
            union = n + cnt[p_] - inter
            iou = inter / (union + 1e-8)
            if cnt[p_] > 0:
                ious_present.append(iou)
            ious_all.append(iou)
        base_miou_present.append(float(np.mean(ious_present)))
        base_miou_all.append(float(np.mean(ious_all)))
    base_acc = float(np.mean(base_acc))
    n_present_mean = float(np.mean(n_present_list))
    print()
    print('--- BASELINE: всегда предсказывать один класс (фон) для всех точек ---')
    print(f'   accuracy                   = {base_acc:.6f}')
    print(f'   mIoU (только присутств.)  = {np.mean(base_miou_present):.6f}   '
          f'<- метрика Train_Hybrid.compute_metrics')
    print(f'   mIoU (все 53 класса)      = {np.mean(base_miou_all):.6f}   '
          f'<- метрика Test_Hybrid.compute_metrics')
    print(f'   ожидаемое отношение acc/mIoU(present) = '
          f'{base_acc / np.mean(base_miou_present):.2f}  (= среднее число присутствующих классов)')
    print('   !! Сравни с training_log: если val_acc/val_mIoU ~= это число,')
    print('      модель выродилась в предсказание одного класса (фона).')

    # ---------------- H4: веса классов ----------------
    print()
    print('=' * 72)
    print('H4. Классовые веса по Eq.2 (w = 1/log(1.02+n), нормировка на 53)')
    print('=' * 72)
    counts = np.where(total_counts == 0, 1, total_counts)
    w = 1.0 / np.log(1.02 + counts)
    w = w / np.sum(w) * NUM_CLASSES
    print(f'   вес фона (класс 0)          : {w[0]:.3f}')
    common = [c for c in range(1, NUM_CLASSES)
              if 0.005 < total_counts[c] / grand < 0.10]
    if common:
        print(f'   вес обычного постоянного зуба : {np.mean([w[c] for c in common]):.3f}')
    rare_c = [c for c in range(1, NUM_CLASSES) if 0 < total_counts[c] / grand < 0.0001]
    if rare_c:
        print(f'   вес редкого зуба            : {np.mean([w[c] for c in rare_c]):.3f}')
    non_bg = [c for c in range(1, NUM_CLASSES) if total_counts[c] > 0]
    med_tooth = float(np.median([total_counts[c] for c in non_bg])) if non_bg else 0.0
    ratio_imbalance = total_counts[0] / max(med_tooth, 1.0)
    ratio_weight = (np.mean([w[c] for c in common]) / w[0]) if common else float('nan')
    print(f'   перекос данных (фон / медианный зуб) : {ratio_imbalance:.1f}x')
    print(f'   перекос в весах (зуб / фон)          : {ratio_weight:.2f}x')
    print(f'   !! Нужно ~ sqrt({ratio_imbalance:.0f}) = {np.sqrt(ratio_imbalance):.1f}x, '
          f'а получается {ratio_weight:.2f}x.')
    print('      Рекомендация: w = 1/sqrt(n) (или (1-n/max)^beta) вместо 1/log(1.02+n).')


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    main()
