import numpy as np
import torch


class ElasticDistortion:
    """Minimal compatible implementation of torch_points3d.core.data_transform."""

    def __init__(self, granularity=(0.2, 0.8), magnitude=(0.4, 1.6),
                 distribution="uniform", apply_distorion=True, **kwargs):
        self.granularity = granularity
        self.magnitude = magnitude
        self.distribution = distribution
        # Сохраняем флаг, чтобы знать, нужно ли применять искажение
        self.apply_distorion = apply_distorion 

    def __call__(self, data):
        # Если флаг False, просто возвращаем данные без изменений
        if not self.apply_distorion:
            return data

        if hasattr(data, "pos"):
            pos = data.pos
            _set = lambda v: setattr(data, "pos", v)
        elif isinstance(data, dict) and "pos" in data:
            pos = data["pos"]
            _set = lambda v: data.__setitem__("pos", v)
        else:
            raise TypeError(
                "ElasticDistortion expects Data with .pos or dict with 'pos'"
            )

        device, dtype = pos.device, pos.dtype
        pos_np = pos.detach().cpu().numpy().astype(np.float32)

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.detach().cpu().numpy()
        else:
            batch = np.zeros(len(pos_np), dtype=np.int64)

        for b in np.unique(batch):
            mask = batch == b
            if mask.sum() < 4:
                continue
            pos_np[mask] = _elastic_warp(
                pos_np[mask], self.granularity, self.magnitude, self.distribution
            )

        _set(torch.from_numpy(pos_np).to(device=device, dtype=dtype))
        return data


def _elastic_warp(points, granularity, magnitude, distribution="uniform"):
    """Apply one random trilinear warp to a batch of points."""
    from scipy.ndimage import map_coordinates

    mn = points.min(axis=0)
    mx = points.max(axis=0)
    scale = np.maximum(mx - mn, 1e-8)
    norm = (points - mn) / scale  

    gran = np.atleast_1d(np.asarray(granularity, dtype=np.float32))
    if gran.size == 1:
        gran = np.repeat(gran, 3)
    grid_shape = tuple(np.ceil(1.0 / gran).astype(int) + 2)

    mag = np.atleast_1d(np.asarray(magnitude, dtype=np.float32))
    mag_val = np.random.uniform(mag.min(), mag.max()) if mag.size > 1 else float(mag[0])

    if distribution == "uniform":
        rand_grid = np.random.uniform(-1, 1, size=(*grid_shape, 3))
    else:
        rand_grid = np.random.normal(0, 1, size=(*grid_shape, 3))

    coords = (norm * (np.array(grid_shape) - 1)).T  
    disp = np.zeros_like(norm)
    for d in range(3):
        disp[:, d] = map_coordinates(rand_grid[..., d], coords, order=1, mode="nearest")

    return (points + disp * mag_val * scale).astype(np.float32)