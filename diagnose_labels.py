#!/usr/bin/env python3
"""
diagnose_labels.py - проверка САМОЙ РАЗМЕТКИ набора 3DTeethSeg'22.

Зачем этот скрипт
-----------------
Мы доказали по per_class_metrics.csv (эпоха 62):
  * 65% точек зубов в validation уходит НЕ в свой класс;
  * 57% из них -> в "background", 43% -> в соседний зуб;
  * средний recall по зубам 0.31 при precision 0.56;
  * все 20 молочных классов (FDI 51-85) в validation имеют НОЛЬ точек.

Из этого следует, что модель плохо различает "поверхность зуба" и "фон", а
также соседние зубы. Есть две принципиально разные причины:

  (A) ПЛОХАЯ РАЗМЕТКА (supervision): порядок вершин в .obj не совпадает с
      порядком меток в .json. Тогда цели "размазаны" по геометрии, и НИКАКОЕ
      обучение не даст резких границ зубов. Модель выучивает только крупные
      формы (что как раз и видно: IoU фона 0.67, у зубов 0.17-0.45).

  (B) ПЛОХОЕ РАЗРЕШЕНИЕ / нехватка данных по редким классам: разметка в
      порядке, но модель не может выделить границы (третьи моляры: IoU~0).

Этот скрипт различает (A) и (B) без обучения - только чтение файлов.

Что он проверяет
----------------
1. count-mismatch: len(вершины .obj) vs len(метки .json). В коде датасета
   стоит min_len = min(features, labels) - МИМОМОЛЧАЩИЙ обрез. Если длины
   не равны, разметка уже частично потеряна.
2. kNN-purity: для каждой точки доля k ближайших соседей с той же меткой.
   При корректной разметке чистота очень высокая (>0.9 у поверхности зуба);
   при "размазанной" разметке она падает до уровня базовой частоты (~0.02-0.1).
3. Компактность классов: разброс точек одного класса относительно размера
   меша. Размазанная разметка -> класс равномерно размазан по всему мешу.
4. Цветовой тест: средний цвет вершин каждого класса. Если метки "съехали",
   средние цвета всех классов совпадают (разброс ~0).
5. Распределение классов (в т.ч. молочные FDI 51-85) по каждому сплиту.
   Это отвечает на вопрос "достижима ли вообще метрика 89.51% из статьи".

Запуск (внутри контейнера HF, CPU, минуты):
    python diagnose_labels.py --root /app/Dataset_900 --splits train validation test \
        --max-cases 20 --knn 10
    python diagnose_labels.py --root /app/Dataset_900 --splits train --max-cases 60 \
        --out per_class_distribution.csv

Зависимости: только numpy (torch НЕ нужен).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

FDI_CODES = ([0] + list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39))
             + list(range(41, 49)) + list(range(51, 56)) + list(range(61, 66))
             + list(range(71, 76)) + list(range(81, 86)))
DECIDUOUS = set(range(51, 56)) | set(range(61, 66)) | set(range(71, 76)) | set(range(81, 86))


def read_obj(path):
    """Вершины и цвета из .obj. Возвращает (xyz, colors, n_floats_per_vertex)."""
    xyz, colors, widths = [], [], defaultdict(int)
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line.startswith('v '):
                continue
            parts = line.split()
            vals = parts[1:]
            widths[len(vals)] += 1
            try:
                xyz.append([float(v) for v in vals[:3]])
                colors.append([float(v) for v in vals[3:6]])
            except ValueError:
                continue
    xyz = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64) if len(colors) else np.zeros((0, 3))
    return xyz, colors, widths


def read_labels(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return np.asarray(data['labels'], dtype=np.int64)


def knn_purity(xyz, labels, k=10, max_points=4000, seed=0):
    """Доля k ближайших соседей с той же меткой (усреднённая по точкам)."""
    n = len(labels)
    if n < k + 2:
        return float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    p = xyz[idx]
    l = labels[idx]
    same = 0
    total = 0
    step = 512
    for s in range(0, len(p), step):
        chunk = p[s:s + step]
        d = ((chunk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d, k + 1, axis=1)[:, 1:k + 1]
        same += int((l[nn] == l[s:s + step, None]).sum())
        total += nn.size
    return same / max(total, 1)


def class_stats(xyz, labels):
    """Компактность и цвет для каждого класса."""
    out = {}
    scale = float(np.max(np.sqrt((xyz - xyz.mean(0)) ** 2).sum(1))) if len(xyz) else 1.0
    for c in np.unique(labels):
        m = labels == c
        if m.sum() == 0:
            continue
        pts = xyz[m]
        ctr = pts.mean(0)
        spread = float(np.sqrt(((pts - ctr) ** 2).sum(1)).mean()) / (scale + 1e-9)
        out[int(c)] = dict(n=int(m.sum()), spread=spread)
    return out


def color_stats(xyz, colors, labels):
    """Разброс средних цветов по классам: если метки разъехались - он ~0."""
    if len(colors) != len(labels) or colors.shape[1] < 3:
        return None
    means = []
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 50:
            continue
        means.append(colors[m].mean(0))
    if len(means) < 2:
        return None
    means = np.asarray(means)
    gmean = colors.mean(0)
    return float(np.sqrt(((means - gmean) ** 2).sum(1)).mean())


def find_cases(obj_root):
    cases = []
    for root, _, files in os.walk(obj_root):
        for f in files:
            if f.endswith('.obj'):
                rel = os.path.relpath(os.path.join(root, f), obj_root)
                cases.append((os.path.join(root, f), rel))
    cases.sort(key=lambda t: t[1])
    return cases


def main():
    ap = argparse.ArgumentParser(description='Диагностика разметки 3DTeethSeg\'22')
    ap.add_argument('--root', default='/app/Dataset_900',
                    help='корень с папками train/validation/test')
    ap.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    ap.add_argument('--max-cases', type=int, default=20,
                    help='сколько случаев разбирать на сплит (0 = все)')
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--out', default='', help='CSV с распределением классов')
    args = ap.parse_args()

    gt_counts = defaultdict(float)   # FDI -> точек (сумма по сплитам)
    rows = []
    jaw_bg_frac = defaultdict(list)   # jaw -> доля фона в кейсе
    jaw_counts = defaultdict(lambda: defaultdict(float))  # jaw -> FDI -> точек

    for split in args.splits:
        obj_dir = os.path.join(args.root, split, 'data')
        gt_dir = os.path.join(args.root, split, 'ground-truth')
        if not os.path.isdir(obj_dir):
            print(f"\n=== {split}: папка {obj_dir} не найдена, пропускаю")
            continue

        cases = find_cases(obj_dir)
        if args.max_cases and args.max_cases < len(cases):
            # детерминированная равномерная выборка по всем кейсам
            step = len(cases) / float(args.max_cases)
            cases = [cases[int(i * step)] for i in range(args.max_cases)]
        print(f"\n=== {split}: всего .obj = {len(find_cases(obj_dir))}, разбираю {len(cases)}")

        mismatch, purities, spreads, color_spreads = [], [], [], []
        split_counts = defaultdict(float)
        n_align_ok = 0

        for obj_path, rel in cases:
            json_path = os.path.join(gt_dir, os.path.splitext(rel)[0] + '.json')
            if not os.path.exists(json_path):
                print(f"  ⚠ нет GT: {rel}")
                continue
            xyz, colors, widths = read_obj(obj_path)
            labels = read_labels(json_path)
            n_v, n_l = len(xyz), len(labels)
            if n_v != n_l:
                mismatch.append((rel, n_v, n_l))

            m = min(n_v, n_l)
            xyz_s, lab_s = xyz[:m], labels[:m]
            valid = (lab_s >= 0) & (lab_s <= 85)
            lab_s = lab_s[valid]
            xyz_s = xyz_s[valid]

            # распределение классов по сырым FDI-кодам
            ids, cnts = np.unique(lab_s, return_counts=True)
            jaw = 'lower' if 'lower' in os.path.basename(obj_path).lower() else (
                'upper' if 'upper' in os.path.basename(obj_path).lower() else '?')
            for i, c in zip(ids, cnts):
                split_counts[int(i)] += int(c)
                gt_counts[int(i)] += int(c)
                jaw_counts[jaw][int(i)] += int(c)
            if len(lab_s):
                jaw_bg_frac[jaw].append(float((lab_s == 0).mean()))

            pur = knn_purity(xyz_s, lab_s, k=args.knn)
            if pur == pur:
                purities.append(pur)
                if pur > 0.5:
                    n_align_ok += 1
            st = class_stats(xyz_s, lab_s)
            if st:
                spreads.append(float(np.mean([v['spread'] for v in st.values()])))
            cs = color_stats(xyz_s, colors[:m][valid] if len(colors) >= m else None, lab_s)
            if cs is not None:
                color_spreads.append(cs)

        print(f"  count-mismatch (len .obj != len .json): {len(mismatch)}/{len(cases)}")
        for rel, nv, nl in mismatch[:5]:
            print(f"      {rel}: obj={nv} json={nl}")
        if purities:
            p = np.asarray(purities)
            print(f"  kNN-purity (k={args.knn}): mean={p.mean():.4f} median={np.median(p):.4f} "
                  f"min={p.min():.4f} max={p.max():.4f}")
            print(f"      кейсов с чистотой >0.5: {n_align_ok}/{len(purities)}")
            print("      >0.5 = метки согласуются с геометрией (разметка ОК);")
            print("      ~0.05 = метки случайны относительно геометрии (разметка СЛОМАНА)")
        if spreads:
            print(f"  компактность классов (средний разброс / радиус меша): {np.mean(spreads):.4f}")
        if color_spreads:
            print(f"  разброс средних цветов по классам: {np.mean(color_spreads):.5f} "
                  f"(≈0 => метки не соответствуют вершинам)")
        dec_pts = sum(v for k, v in split_counts.items() if k in DECIDUOUS)
        tot = sum(split_counts.values())
        print(f"  точек всего: {int(tot):,} | молочных (FDI 51-85): {int(dec_pts):,} "
              f"({100 * dec_pts / max(tot, 1):.3f}%)")
        present = sorted(split_counts.keys())
        print(f"  классов с ненулевым GT: {len(present)} из 53 "
              f"(отсутствуют: {sorted(set(FDI_CODES) - set(present))})")

        if jaw_bg_frac:
            print("  по челюстям:")
            for jaw in sorted(jaw_bg_frac):
                fr = np.asarray(jaw_bg_frac[jaw])
                jc = jaw_counts[jaw]
                tot_j = sum(jc.values()) or 1
                n_teeth = len([k for k, v in jc.items() if k != 0 and v > 0])
                print(f"      {jaw:5s}: кейсов {len(fr):3d}, доля фона {fr.mean():.3f} "
                      f"(мин {fr.min():.3f}, макс {fr.max():.3f}), "
                      f"зубов с GT {n_teeth}, доля фона в точках {100 * jc.get(0, 0) / tot_j:.1f}%")

    print("\n=== СВОДКА: точки по FDI-кодам (все разобранные сплиты) ===")
    for c in FDI_CODES:
        v = gt_counts.get(c, 0)
        mark = ' (молочный)' if c in DECIDUOUS else ''
        if v == 0:
            print(f"  FDI {c:2d}: 0{mark}")
    if args.out:
        with open(args.out, 'w', newline='') as f:
            f.write('fdi_code,is_deciduous,gt_points\n')
            for c in FDI_CODES:
                f.write(f"{c},{1 if c in DECIDUOUS else 0},{int(gt_counts.get(c, 0))}\n")
        print(f"\nCSV -> {args.out}")

    print("\nГотово.")


if __name__ == '__main__':
    sys.exit(main())
