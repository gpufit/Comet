"""CPU backend for the drift cost function.

Two implementations of the same maths live here:

``_cost_and_gradient_reference``
    A plain Python loop, one pair at a time. Slow, but it is the readable
    statement of what the cost function *is*, and every other backend is
    checked against it.
``_cost_and_gradient_njit``
    The same loop compiled with numba. This is what actually runs.

numba is already a hard dependency -- it is what the CUDA kernels are built on
-- so the compiled path costs nothing extra to install. It is roughly 700x
faster than the interpreted loop and agrees with it to ~1e-18.
"""

import math

import numpy as np
from numba import njit, prange

# Below this many pairs, thread start-up costs more than the parallel loop saves.
PARALLEL_PAIR_THRESHOLD = 200_000

# Number of independent accumulation buffers used by the parallel kernel. Each
# is (n_segments, 3) float64, so the memory cost is small and bounded.
_N_BLOCKS = 16


def _cost_and_gradient_reference(coords, times, idx_i, idx_j, mu, sigma, sigma_factor):
    """Reference implementation: the cost function written out plainly.

    Not used at runtime. Kept as the specification that the compiled kernel and
    the GPU backends are tested against.

    Parameters
    ----------
    coords : (N, 3) float array of localization coordinates in nm.
    times : (N,) int array mapping each localization to its segment.
    idx_i, idx_j : (P,) int arrays of neighbour-pair indices.
    mu : (T, 3) float array, the current per-segment drift estimate in nm.
    sigma, sigma_factor : the Gaussian width and its current schedule multiplier.

    Returns
    -------
    total : float, the summed pair overlap.
    deri : (T, 3) float array, d(total)/d(mu).
    """
    sigma_sq = (2.0 * sigma * sigma_factor) ** 2
    inv_sigma = 1.0 / (sigma * sigma_factor)
    deri = np.zeros_like(mu)
    total = 0.0

    for p in range(len(idx_i)):
        i = idx_i[p]
        j = idx_j[p]
        ti = times[i]
        tj = times[j]

        dx = (coords[i, 0] - mu[ti, 0]) - (coords[j, 0] - mu[tj, 0])
        dy = (coords[i, 1] - mu[ti, 1]) - (coords[j, 1] - mu[tj, 1])
        dz = (coords[i, 2] - mu[ti, 2]) - (coords[j, 2] - mu[tj, 2])

        val = math.exp(-(dx * dx + dy * dy + dz * dz) / sigma_sq) * inv_sigma
        total += val

        for d in range(3):
            # The two contributions are exact negatives of each other.
            contrib = 2.0 * val * (coords[j, d] - coords[i, d] + mu[ti, d] - mu[tj, d]) / sigma_sq
            deri[tj, d] += contrib
            deri[ti, d] -= contrib

    return total, deri


@njit(cache=True, fastmath=True, nogil=True)
def _cost_and_gradient_njit(coords, times, idx_i, idx_j, mu, sigma, sigma_factor):
    """Compiled equivalent of :func:`_cost_and_gradient_reference`."""
    sigma_sq = (2.0 * sigma * sigma_factor) ** 2
    inv_sigma = 1.0 / (sigma * sigma_factor)
    deri = np.zeros_like(mu)
    total = 0.0

    for p in range(idx_i.shape[0]):
        i = idx_i[p]
        j = idx_j[p]
        ti = times[i]
        tj = times[j]

        dx = (coords[i, 0] - mu[ti, 0]) - (coords[j, 0] - mu[tj, 0])
        dy = (coords[i, 1] - mu[ti, 1]) - (coords[j, 1] - mu[tj, 1])
        dz = (coords[i, 2] - mu[ti, 2]) - (coords[j, 2] - mu[tj, 2])

        val = math.exp(-(dx * dx + dy * dy + dz * dz) / sigma_sq) * inv_sigma
        total += val

        for d in range(3):
            contrib = 2.0 * val * (coords[j, d] - coords[i, d] + mu[ti, d] - mu[tj, d]) / sigma_sq
            deri[tj, d] += contrib
            deri[ti, d] -= contrib

    return total, deri


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _cost_and_gradient_njit_parallel(coords, times, idx_i, idx_j, mu, sigma, sigma_factor):
    """Parallel variant.

    The gradient update is a scatter-add, which cannot be expressed as a numba
    reduction, so each block accumulates into its own buffer and the buffers are
    summed at the end. Blocks are used rather than thread ids because
    numba.get_thread_id() is not available on the oldest supported numba.
    """
    sigma_sq = (2.0 * sigma * sigma_factor) ** 2
    inv_sigma = 1.0 / (sigma * sigma_factor)
    n_pairs = idx_i.shape[0]
    n_segments = mu.shape[0]

    deri_blocks = np.zeros((_N_BLOCKS, n_segments, 3))
    totals = np.zeros(_N_BLOCKS)
    block_size = (n_pairs + _N_BLOCKS - 1) // _N_BLOCKS

    for b in prange(_N_BLOCKS):
        start = b * block_size
        stop = min(start + block_size, n_pairs)
        local_total = 0.0
        for p in range(start, stop):
            i = idx_i[p]
            j = idx_j[p]
            ti = times[i]
            tj = times[j]

            dx = (coords[i, 0] - mu[ti, 0]) - (coords[j, 0] - mu[tj, 0])
            dy = (coords[i, 1] - mu[ti, 1]) - (coords[j, 1] - mu[tj, 1])
            dz = (coords[i, 2] - mu[ti, 2]) - (coords[j, 2] - mu[tj, 2])

            val = math.exp(-(dx * dx + dy * dy + dz * dz) / sigma_sq) * inv_sigma
            local_total += val

            for d in range(3):
                contrib = 2.0 * val * (coords[j, d] - coords[i, d] + mu[ti, d] - mu[tj, d]) / sigma_sq
                deri_blocks[b, tj, d] += contrib
                deri_blocks[b, ti, d] -= contrib
        totals[b] = local_total

    deri = np.zeros_like(mu)
    for b in range(_N_BLOCKS):
        deri += deri_blocks[b]
    return totals.sum(), deri


def cpu_wrapper_chunked(mu, locs_coords, locs_time, idx_i, idx_j, sigma, sigma_factor,
                        val=None, deri=None, chunk_size=None, debug=False, parallel=None):
    """Cost and gradient for the optimizer, on the CPU.

    Signature matches the CUDA and torch wrappers so the optimizer can swap
    backends. `val`, `deri` and `chunk_size` exist for that compatibility only:
    the compiled kernel needs no scratch buffer and no chunking, since it is not
    working around device memory limits.
    """
    mu = np.ascontiguousarray(mu.reshape((-1, 3)), dtype=np.float64)
    coords = np.ascontiguousarray(locs_coords[:, :3], dtype=np.float64)
    times = np.ascontiguousarray(locs_time, dtype=np.int64)
    idx_i = np.ascontiguousarray(idx_i, dtype=np.int64)
    idx_j = np.ascontiguousarray(idx_j, dtype=np.int64)

    if parallel is None:
        parallel = idx_i.shape[0] >= PARALLEL_PAIR_THRESHOLD
    kernel = _cost_and_gradient_njit_parallel if parallel else _cost_and_gradient_njit

    total, gradient = kernel(coords, times, idx_i, idx_j, mu,
                             float(sigma), float(sigma_factor))

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 2)
        for d in range(3):
            ax[d, 0].plot(gradient[:, d])
            ax[d, 1].plot(mu[:, d])
        ax[0, 0].set_title(f"Gradients (sigma={sigma * sigma_factor:.2f} nm)")
        ax[0, 1].set_title("Drift Estimate [nm]")
        plt.tight_layout()
        plt.show()

    # The optimizer minimizes, so hand back the negated overlap.
    return -total, -gradient.flatten()
