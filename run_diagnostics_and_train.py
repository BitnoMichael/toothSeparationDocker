#!/usr/bin/env python3
"""
run_diagnostics_and_train.py - один запуск: сначала проверка ДАННЫХ, потом обучение.

Зачем
----
`diagnose_labels.py` отвечает на главный открытый вопрос: не сломана ли сама
разметка 3DTeethSeg'22 (совпадает ли порядок вершин .obj с порядком меток .json).
Это проверяется за минуты на CPU, без обучения, и от ответа зависит, стоит ли
вообще продолжать текущий прогон.

Скрипт делает:
  1) диагностика разметки по всем сплитам -> /app/checkpoints/diag/
  2) проверка утверждения про веса классов (verify_class_weights.py), если рядом
     лежит per_class_metrics.csv текущего прогона
  3) запуск Train_Hybrid.py с переданными аргументами (resume и т.д.)

Если диагностика падает - обучение всё равно запускается (ошибка не должна
убивать прогон).

Запуск (внутри контейнера):
    python run_diagnostics_and_train.py --diag_max_cases 15 -- \
        Train_Hybrid.py --save_path /app/checkpoints/run2 --resume_from last \
        --early_stopping_patience 40 --bn_momentum 0.01 --bn_recalibrate 40 \
        --dump_per_class

Аргументы после "--" уходят в Train_Hybrid.py как есть.
Альтернатива: --train_args "Train_Hybrid.py --save_path ..." (одна строка).
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Те же аргументы, что и в рабочей команде запуска. Используются, если в
# argv не нашлось явной команды обучения (например, если CLI обрезал "--"):
# так прогон всё равно продолжит обучение с теми же настройками, а не
# молча стартует с нуля на дефолтных путях.
DEFAULT_TRAIN_ARGS = [
    'Train_Hybrid.py',
    '--save_path', '/app/checkpoints/run2',
    '--resume_from', 'last',
    '--early_stopping_patience', '40',
    '--bn_momentum', '0.01',
    '--bn_recalibrate', '40',
    '--dump_per_class',
]


def run(cmd, log_path=None):
    """Запуск команды с одновременной записью в stdout и в файл."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
            proc.wait()
        return proc.returncode
    return subprocess.call(cmd)


def main():
    argv = sys.argv[1:]
    train_args = []
    if '--' in argv:
        i = argv.index('--')
        diag_argv, train_args = argv[:i], argv[i + 1:]
    else:
        # Разделитель мог срезаться в CLI. Тогда ищем первый аргумент,
        # похожий на скрипт, и всё после него считаем командой обучения.
        diag_argv = argv
        for j, a in enumerate(argv):
            if a.endswith('.py') and not a.startswith('-'):
                diag_argv, train_args = argv[:j], argv[j:]
                print(f"(разделитель '--' не дошёл - разбираю argv сами: "
                      f"диагностика={diag_argv}, обучение={train_args})")
                break

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--diag_root', default='/app/Dataset_900')
    ap.add_argument('--diag_splits', nargs='+',
                    default=['train', 'validation', 'test'])
    ap.add_argument('--diag_max_cases', type=int, default=15,
                    help='кейсов на сплит (0 = все)')
    ap.add_argument('--diag_knn', type=int, default=10)
    ap.add_argument('--diag_out_dir', default='/app/checkpoints/diag')
    ap.add_argument('--skip_diag', action='store_true')
    ap.add_argument('--train_args', default='',
                    help='строка с командой обучения (альтернатива "--")')
    args, unknown = ap.parse_known_args(diag_argv)

    if args.train_args:
        train_args = shlex.split(args.train_args)
    if not train_args:
        train_args = list(DEFAULT_TRAIN_ARGS)
        print("⚠️ Команда обучения не распознана - использую рабочий дефолт:\n   " +
              " ".join(DEFAULT_TRAIN_ARGS))
    if unknown:
        print(f"⚠️ Неизвестные аргументы диагностики (игнорирую): {unknown}")

    os.makedirs(args.diag_out_dir, exist_ok=True)
    t0 = time.time()
    summary_path = os.path.join(args.diag_out_dir, 'SUMMARY.txt')

    def write_summary(lines):
        try:
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            print(f"(не смог записать {summary_path}: {e})")

    write_summary(["=" * 60,
                   f"Запуск: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                   f"Команда обучения: {' '.join(train_args)}",
                   "=" * 60])

    if not args.skip_diag:
        print("=" * 70)
        print("ШАГ 1/2: диагностика разметки и распределения классов")
        print("=" * 70, flush=True)
        csv_out = os.path.join(args.diag_out_dir, 'per_class_distribution.csv')
        cmd = [sys.executable, os.path.join(HERE, 'diagnose_labels.py'),
               '--root', args.diag_root,
               '--splits'] + list(args.diag_splits) + [
               '--max-cases', str(args.diag_max_cases),
               '--knn', str(args.diag_knn),
               '--out', csv_out]
        rc = run(cmd, os.path.join(args.diag_out_dir, 'diagnose_labels.log'))
        print(f"[диагностика] rc={rc}, время {time.time() - t0:.0f} с")
        if rc != 0:
            print("⚠️ Диагностика завершилась с ошибкой - продолжаю обучение.")
        # Копия ключевых строк в сводку (на случай, если лог большой)
        log_path = os.path.join(args.diag_out_dir, 'diagnose_labels.log')
        if os.path.exists(log_path):
            with open(log_path, encoding='utf-8') as f:
                txt = f.read()
            keys = [l for l in txt.splitlines()
                    if ('mismatch' in l or 'purity' in l or
                        ('точек всего' in l and 'молочн' in l) or
                        'классов с ненулевым' in l or 'по челюстям' in l or
                        'значений на вершину' in l or
                        l.strip().startswith(('upper', 'lower')))]
            write_summary(["--- ДИАГНОСТИКА РАЗМЕТКИ ---"] + keys)

        # Проверка утверждения про веса классов (нужен per_class_metrics.csv)
        print("\n--- проверка утверждения про веса классов ---", flush=True)
        save_path = None
        idx = train_args.index('--save_path') if '--save_path' in train_args else -1
        if idx >= 0 and idx + 1 < len(train_args):
            save_path = train_args[idx + 1]
        csv_candidates = []
        if save_path:
            csv_candidates.append(os.path.join(save_path, 'per_class_metrics.csv'))
        csv_candidates.append(os.path.join(args.diag_out_dir, 'per_class_metrics.csv'))
        csv_candidates.append('per_class_metrics.csv')
        csv_in = next((c for c in csv_candidates if os.path.exists(c)), None)
        if csv_in:
            run([sys.executable, os.path.join(HERE, 'verify_class_weights.py'),
                 '--csv', csv_in],
                os.path.join(args.diag_out_dir, 'verify_class_weights.log'))
        else:
            print("(per_class_metrics.csv не найден - проверка весов пропущена)")

    # Скрипт обучения ищем рядом с этим файлом, если его нет в текущем каталоге
    script = train_args[0]
    if not os.path.exists(script) and os.path.exists(os.path.join(HERE, script)):
        train_args = [os.path.join(HERE, script)] + train_args[1:]

    print("\n" + "=" * 70)
    print("ШАГ 2/2: обучение")
    print("=" * 70, flush=True)
    rc = run([sys.executable] + train_args)
    print(f"\nОбучение завершилось с кодом {rc} (время всего {time.time() - t0:.0f} с)")
    return rc


if __name__ == '__main__':
    sys.exit(main())
