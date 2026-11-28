#!/usr/bin/env python3
"""
verify_class_weights.py - проверка утверждения "веса пустых классов не портят loss".

Утверждение (проверено на реальных данных, эпоха 62, per_class_metrics.csv):
  разница между режимами 'legacy' и 'present' весов классов сводится к
  ПОСТОЯННОМУ множителю для всех непустых классов, потому что FocalLoss в
  Train_Hybrid.py использует focal_loss.mean() (обычное среднее по точкам),
  а вес берётся по истинной метке точки (alpha[targets]). Пустые классы не
  имеют целевых точек -> их вес не входит ни в числитель, ни в знаменатель
  loss. При AdamW (инвариантен к масштабу градиента) это не меняет обучение.

Скрипт:
  * берёт реальные количества точек по 53 классам (из per_class_metrics.csv,
    если он передан, иначе - из встроенного снимка эпохи 62);
  * считает веса в обоих режимах;
  * проверяет, что отношение w_present/w_legacy одинаково для всех живых
    классов (и печатает коэффициент);
  * считает реальную долю loss (mass ~ w_c * n_c) для пустых классов.

Запуск:  python verify_class_weights.py [--csv per_class_metrics.csv]
"""

import argparse
import csv
import os

import numpy as np

NUM_CLASSES = 53

# Снимок реального распределения: epoch 62, per_class_metrics.csv
# (class_idx -> {gt_points, fdi_code})
SNAPSHOT = {
    0: (10445203, 0), 1: (457936, 11), 2: (317261, 12), 3: (320911, 13),
    4: (410295, 14), 5: (433854, 15), 6: (677475, 16), 7: (354247, 17),
    8: (19132, 18), 9: (468666, 21), 10: (328707, 22), 11: (328550, 23),
    12: (411208, 24), 13: (421350, 25), 14: (688959, 26), 15: (368242, 27),
    16: (21317, 28), 17: (351948, 31), 18: (377917, 32), 19: (420220, 33),
    20: (471170, 34), 21: (541631, 35), 22: (867299, 36), 23: (497387, 37),
    24: (28857, 38), 25: (350170, 41), 26: (378603, 42), 27: (416922, 43),
    28: (482968, 44), 29: (540759, 45), 30: (866947, 46), 31: (496464, 47),
    32: (30385, 48),
}


def load_counts(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        print(f"(файл {csv_path!r} не найден - беру встроенный снимок эпохи 62)")
        n = np.zeros(NUM_CLASSES)
        for i, (gt, _) in SNAPSHOT.items():
            n[i] = gt
        return n
    n = np.zeros(NUM_CLASSES)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            n[int(row['class_idx'])] = int(row['gt_points'])
    print(f"Counts loaded from {csv_path}")
    return n


def weights(counts, mode):
    absent = counts == 0
    c = np.where(absent, 0.0 if mode == 'present' else 1.0, counts)
    raw = np.zeros(NUM_CLASSES)
    nz = c > 0
    raw[nz] = 1.0 / np.log(1.02 + c[nz])
    tot = raw.sum()
    return raw / tot * NUM_CLASSES if tot else np.ones(NUM_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='per_class_metrics.csv')
    args = ap.parse_args()

    counts = load_counts(args.csv)
    absent = counts == 0
    wl = weights(counts, 'legacy')
    wp = weights(counts, 'present')

    print(f"Классов без единой точки: {int(absent.sum())} из {NUM_CLASSES}")
    print(f"Доля СУММЫ ВЕСОВ у пустых классов (legacy): {100 * wl[absent].sum() / wl.sum():.1f}%")
    mass_l = wl * counts
    mass_p = wp * counts
    print(f"Доля LOSS у пустых классов (legacy): {100 * mass_l[absent].sum() / mass_l.sum():.4f}%")
    print(f"Доля LOSS у пустых классов (present): {100 * mass_p[absent].sum() / mass_p.sum():.4f}%")

    live = ~absent
    ratio = wp[live] / wl[live]
    print(f"Отношение w_present/w_legacy по {live.sum()} живым классам: "
          f"min={ratio.min():.6f} max={ratio.max():.6f}")
    if abs(ratio.max() - ratio.min()) < 1e-9:
        print("  -> КОНСТАНТА: относительные градиенты идентичны, разница только в масштабе.")
        print("     AdamW инвариантен к масштабу градиента => обучение не меняется.")
    else:
        print("  -> НЕ константа: относительные градиенты различаются, проверьте вручную.")

    print(f"Фон: доля GT {100 * counts[0] / counts.sum():.1f}%, "
          f"доля LOSS {100 * mass_l[0] / mass_l.sum():.1f}%, вес {wl[0]:.4f}")
    print("Готово.")


if __name__ == '__main__':
    main()
