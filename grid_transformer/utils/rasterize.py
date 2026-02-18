

import numpy as np

import numpy as np

def _grid_hw(Lx, Ly, pixel_size):
    W = int(np.floor(Lx / pixel_size))
    H = int(np.floor(Ly / pixel_size))
    if H <= 0 or W <= 0:
        raise ValueError("H or W became non-positive; check L and pixel_size.")
    return H, W

def rasterize_binary(coords, Lx, Ly, pixel_size, wrap=True):
    """
    Single snapshot → binary grid.

    Args
    ----
    coords: (N,2) float array of particle centers (same units as L)
    L:      (2,)  float array [Lx, Ly]
    pixel_size: float, pixel size in same units as coords/L
    wrap:   if True, apply periodic wrap into [0, L)

    Returns
    -------
    grid: (H, W) float32, 1 where a center falls into that pixel, else 0
    """
    coords = np.asarray(coords, dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N,2), got {coords.shape}")

    if wrap:
        coords = coords.copy()
        coords[:, 0] = np.mod(coords[:, 0], Lx)
        coords[:, 1] = np.mod(coords[:, 1], Ly)

    H, W = _grid_hw(Lx, Ly, pixel_size)
    grid = np.zeros((H, W), dtype=np.float32)
    if coords.size == 0:
        return grid

    x_idx = (coords[:, 0] / pixel_size).astype(int) % W
    y_idx = (coords[:, 1] / pixel_size).astype(int) % H
    grid[y_idx, x_idx] = 1.0
    return grid

def rasterize_binary_batch(coords_b, L_b, pixel_size, wrap=True):
    """
    Batched rasterization.

    Args
    ----
    coords_b: (B,N,2) array OR list of (Ni,2) arrays (supports variable N)
    L_b:      (B,2) array OR list of (2,) arrays; per-sample boxes
    pixel_size: float
    wrap:     periodic wrap per sample

    Returns
    -------
    grids: list of (Hi, Wi) float32 arrays (sizes may differ if L differs)
    """
    # Normalize to lists so we can handle variable N and variable L
    if isinstance(coords_b, np.ndarray) and coords_b.ndim == 3:
        coords_list = [coords_b[i] for i in range(coords_b.shape[0])]
    elif isinstance(coords_b, (list, tuple)):
        coords_list = list(coords_b)
    else:
        raise ValueError("coords_b must be (B,N,2) array or list of (Ni,2) arrays.")

    if isinstance(L_b, np.ndarray) and L_b.ndim == 2:
        L_list = [L_b[i] for i in range(L_b.shape[0])]
    elif isinstance(L_b, (list, tuple)):
        L_list = list(L_b)
    else:
        raise ValueError("L_b must be (B,2) array or list of (2,) arrays.")

    if len(coords_list) != len(L_list):
        raise ValueError("coords_b and L_b must have the same length/B dimension.")

    grids = []
    for coords, L in zip(coords_list, L_list):
        grids.append(rasterize_binary(coords, L, pixel_size, wrap=wrap))
    return grids


def random_cyclic_roll(arr, rng):
    """Random periodic shift for translation equivariance."""
    H, W = arr.shape[-2], arr.shape[-1]
    di = rng.integers(0, H) if H > 1 else 0
    dj = rng.integers(0, W) if W > 1 else 0
    return np.roll(np.roll(arr, di, axis=-2), dj, axis=-1), (di, dj)
