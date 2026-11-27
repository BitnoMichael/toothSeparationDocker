"""
ElasticDistortion для пайплайна 3DTeethSeg (53 класса).

Зачем этот файл
---------------
1) В репозитории лежала папка `torch_points3d/` с самописной реализацией
   ElasticDistortion. Локальная папка ПЕРЕКРЫВАЕТ установленный пакет
   `torch-points3d` (Python находит её раньше site-packages), поэтому
   поставить настоящий пакет было невозможно, а реализация была неверной.

2) Семантика upstream (документация torch-points3d, класс ElasticDistortion):
       "First projects the position onto a voxel grid and then apply the
        distortion to the voxel grid"
       granularity - "Granularity of the noise in meters"  -> РАЗМЕР ЯЧЕЙКИ
                     шума в единицах координат (у нас это мм)
       magnitude   - "Noise multiplier in meters"          -> АМПЛИТУДА
                     смещения в единицах координат (мм)
   Итоговое смещение = interp(coords) * magnitude, БЕЗ умножения на размер bbox.

3) Ошибка была ровно в этом множителе. Самописный вариант:
       return points + disp * mag_val * scale      # scale = размах bbox, мм
   На сканах 3DTeethSeg (размах 60-80 мм) magnitude 0.4-1.6 давал смещение
   до 100 мм - зубы "разъезжались", локальная геометрия разрушалась, и модель
   могла выучить только априорный класс (фон). Это и было причиной
   val accuracy ~44% (= доля фона) и val mIoU ~0.031 (= доля фона / 13.99).

Как пользоваться
----------------
    from utils.elastic_distortion import ElasticDistortion
    ed = ElasticDistortion(apply_distorsion=True,
                           granularity=[0.2, 0.8],   # мм
                           magnitude=[0.4, 1.6])     # мм -> смещение ~0.2-0.5 мм
    data = ed(data)

Режимы
------
* По умолчанию используется локальный порт той же математики: чистый numpy,
  без scipy, без тяжёлой сборки сетки шума.
* Если установлен настоящий torch-points3d и задан TP3D_USE_UPSTREAM=1,
  используется он (с проверкой формы на выходе и откатом на порт при сбое).
* Папку `torch_points3d/` из репозитория нужно УДАЛИТЬ - иначе она
  продолжает перекрывать установленный пакет.
"""
import os
import numpy as np

try:
    import torch
except ImportError:  # numpy-ядро работает и без torch (тесты, анализ)
    torch = None

# Порядок величин сетки шума: не даём памяти уйти в гигабайты на сэмпл.
# РАЗРЕШЕНО вручную (2026-11-26): конфликт слияния E4/E5 разрушал файл.
# Взята векторизованная реализация из E4, но с MAX_GRID_CELLS = 128.
#
# Почему 128, а не 64: upstream torch-points3d строит сетку шума как
#   noise_dim = ceil(bbox / granularity) + 3
# т.е. для скана 3DTeethSeg (размах ~70 мм) при granularity = 0.2-0.8 мм
# получается ~90-350 ячеек по оси. Значение 128 соответствует середине этого
# диапазона и даёт eff_gran = extent/128 ~ 0.55 мм - то есть ту же частоту
# поля шума, что и upstream. При 64 ячейках eff_gran ~ 1.1 мм, шум становится
# вдвое-вчетверо более крупномасштабным, т.е. искажение слабее, чем в статье.
#
# Замер на 100k точек (сканы 3DTeethSeg): 128^3 -> 440 мс/сэмпл (+6.5 мин к
# каждой эпохе), 64^3 -> 103 мс/сэмпл (+1.5 мин к эпохе). Амплитуда смещения
# при этом не меняется - меняется только частота поля (размер ячейки шума).
MAX_GRID_CELLS = 128

# Оригинальные параметры статьи трактуются в миллиметрах (единицы координат):
# granularity -> размер ячейки шума, magnitude -> амплитуда смещения.
DEFAULT_GRANULARITY = (0.2, 0.8)
DEFAULT_MAGNITUDE = (0.4, 1.6)


# ---------------------------------------------------------------- upstream
def _load_upstream():
    """Вернуть настоящий torch_points3d.ElasticDistortion, если он установлен.

    Локальный шим внутри репозитория игнорируется: он перекрывает пакет,
    но именно его мы и чиним.
    """
    if os.environ.get("TP3D_USE_UPSTREAM", "0") not in ("1", "true", "True"):
        return None
    try:
        import importlib
        mod = importlib.import_module("torch_points3d.core.data_transform")
    except Exception:
        return None
    path = getattr(mod, "__file__", "") or ""
    norm = path.replace("\\", "/")
    if "/site-packages/" in norm or "/dist-packages/" in norm:
        return getattr(mod, "ElasticDistortion", None)
    return None


# ------------------------------------------------------------- numpy-порт
def _smooth_axis(arr, axis):
    """Сглаживание ядром [1,2,1]/4 вдоль оси, падение константой 0 (cval=0).

    Считаем без np.concatenate (копия массива 128^3 x 3 на каждой оси стоила
    больше, чем сама интерполяция) и без np.moveaxis, чтобы результат всегда
    оставался C-контигуосным: иначе последующий reshape() молча делал бы
    ещё одну полную копию.
    """
    out = np.empty_like(arr)

    def sl(start, stop):
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return tuple(idx)

    # внутренние отсчёты: (prev + 2*cur + next) / 4
    out[sl(1, -1)] = (arr[sl(None, -2)] + 2.0 * arr[sl(1, -1)] + arr[sl(2, None)]) * 0.25
    # края: за границей - нулевая паддинг-строка
    out[sl(0, 1)] = (2.0 * arr[sl(0, 1)] + arr[sl(1, 2)]) * 0.25
    out[sl(-1, None)] = (arr[sl(-2, -1)] + 2.0 * arr[sl(-1, None)]) * 0.25
    return out


def _trilinear_sample(grid, coords):
    """Трилинейная интерполяция сразу по всем каналам.

    grid  : (nx, ny, nz, C) - значения каналов
    coords: (3, N) - индексы в сетке (дробные)
    Возвращает (N, C).

    Сетка хранится вплотивную, поэтому на каждую из 8 октантов приходится
    ОДИН gather на (N, C), а не по одному на каждый канал отдельно.
    """
    nx, ny, nz = grid.shape[:3]
    c = [np.clip(coords[d], 0.0, (nx, ny, nz)[d] - 1.0 - 1e-6) for d in range(3)]
    i0 = [np.floor(c[d]).astype(np.intp) for d in range(3)]
    f = [c[d] - i0[d] for d in range(3)]
    flat = grid.reshape(-1, grid.shape[3])
    out = np.zeros((coords.shape[1], grid.shape[3]), dtype=grid.dtype)
    for bx in (0, 1):
        wx = f[0] if bx else (1.0 - f[0])
        for by in (0, 1):
            wy = f[1] if by else (1.0 - f[1])
            for bz in (0, 1):
                wz = f[2] if bz else (1.0 - f[2])
                idx = ((i0[0] + bx) * ny + (i0[1] + by)) * nz + (i0[2] + bz)
                out += (wx * wy * wz)[:, None] * flat[idx]
    return out


def elastic_distortion_np(points, granularity=DEFAULT_GRANULARITY,
                           magnitude=DEFAULT_MAGNITUDE, distribution="gaussian"):
    """Канонический ElasticDistortion (семантика torch-points3d) на чистом numpy.

    points      : (N,3) float
    granularity : размер ячейки шума в единицах координат (мм)
    magnitude   : амплитуда смещения в единицах координат (мм)
    """
    gran = np.atleast_1d(np.asarray(granularity, dtype=np.float64))
    gran_val = float(np.random.uniform(gran.min(), gran.max())) if gran.size > 1 else float(gran[0])
    mag = np.atleast_1d(np.asarray(magnitude, dtype=np.float64))
    mag_val = float(np.random.uniform(mag.min(), mag.max())) if mag.size > 1 else float(mag[0])
    gran_val = max(gran_val, 1e-6)

    coords_min = points.min(axis=0)
    extent = (points - coords_min).max(axis=0)

    # Ячейка не может быть меньше extent/MAX_GRID_CELLS, иначе сетка шума
    # раздувается до сотен миллионов ячеек (именно это и ломало оригинал).
    eff_gran = max(gran_val, float(extent.max()) / MAX_GRID_CELLS)

    noise_dim = np.maximum((extent / eff_gran).astype(np.int64) + 3, 3)
    if distribution == "uniform":
        noise = np.random.uniform(-1.0, 1.0, size=tuple(noise_dim) + (3,)).astype(np.float32)
    else:
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    for axis in range(3):
        noise = _smooth_axis(noise, axis)

    ax_min = coords_min - eff_gran
    ax_max = coords_min + eff_gran * (noise_dim - 2)
    coords = np.empty((points.shape[0], 3), dtype=np.float32)
    for d in range(3):
        span = max(ax_max[d] - ax_min[d], 1e-8)
        coords[:, d] = (points[:, d] - ax_min[d]) * (noise_dim[d] - 1) / span

    # (N, C) - все три компоненты смещения за один проход по сетке.
    disp = _trilinear_sample(noise, coords.T)

    # magnitude - это амплитуда в мм, а НЕ множитель размера bounding box.
    return (points + disp * mag_val).astype(np.float32)


class ElasticDistortion:
    """Drop-in замена torch_points3d.core.data_transform.ElasticDistortion.

    API совместим: apply_distorsion / granularity / magnitude / distribution,
    __call__(data) -> data (data.pos обновляется на месте).
    """

    def __init__(self, granularity=DEFAULT_GRANULARITY, magnitude=DEFAULT_MAGNITUDE,
                 distribution="gaussian", apply_distorsion=True,
                 apply_distorion=None, **kwargs):
        self.granularity = granularity
        self.magnitude = magnitude
        self.distribution = distribution
        if apply_distorion is not None:      # старая опечатка в вызывающем коде
            apply_distorsion = apply_distorion
        self.apply_distorsion = apply_distorsion
        self._upstream = _load_upstream()
        self._warned = False

    # -- helpers ---------------------------------------------------------
    def _apply_port(self, pos_np):
        if hasattr(self, "_batch") and self._batch is not None:
            out = pos_np.copy()
            for b in np.unique(self._batch):
                mask = self._batch == b
                if mask.sum() < 4:
                    continue
                out[mask] = elastic_distortion_np(
                    pos_np[mask], self.granularity, self.magnitude, self.distribution)
            return out
        return elastic_distortion_np(pos_np, self.granularity, self.magnitude,
                                     self.distribution)

    def _apply_upstream(self, data):
        """Вызов настоящего пакета. Возвращает None, если он непригоден."""
        try:
            from torch_geometric.data import Data
            src = data.pos.detach().cpu().numpy().astype(np.float32)
            obj = Data(pos=torch.from_numpy(src))
            out = self._upstream(obj)
            res = out.pos.detach().cpu().numpy()
            if res.shape != src.shape or not np.isfinite(res).all():
                return None
            return res.astype(np.float32)
        except Exception:
            return None

    # -- main ------------------------------------------------------------
    def __call__(self, data):
        if not self.apply_distorsion:
            return data

        if hasattr(data, "pos"):
            pos = data.pos
            _set = lambda v: setattr(data, "pos", v)
        elif isinstance(data, dict) and "pos" in data:
            pos = data["pos"]
            _set = lambda v: data.__setitem__("pos", v)
        else:
            raise TypeError("ElasticDistortion expects Data with .pos or dict with 'pos'")

        device, dtype = pos.device, pos.dtype
        pos_np = pos.detach().cpu().numpy().astype(np.float32)

        self._batch = None
        if hasattr(data, "batch") and data.batch is not None:
            self._batch = data.batch.detach().cpu().numpy()

        result = None
        if self._upstream is not None:
            result = self._apply_upstream(data)
            if result is None and not self._warned:
                print("Warning: upstream torch_points3d ElasticDistortion returned an "
                      "unexpected shape -> falling back to the local numpy port.")
                self._warned = True

        if result is None:
            result = self._apply_port(pos_np)

        _set(self._to_tensor_like(result, pos, device, dtype))
        return data

    @staticmethod
    def _to_tensor_like(arr, ref, device, dtype):
        """Вернуть результат тем же типом, что и входной pos."""
        if torch is not None and hasattr(ref, "detach"):
            return torch.from_numpy(arr).to(device=device, dtype=dtype)
        return arr
