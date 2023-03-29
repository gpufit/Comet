import math
import numpy as np
from numba import cuda
from scipy.optimize import minimize
from pair_indices import cuda_wrapper_pair_indices, pair_indices_simplified, pair_indices_lex_floor, pair_indices_lex


@cuda.jit
def cost_function_full_2d_chunked(d_locs_time, start_idx, chunk_size, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val,
                                  d_val_sum, d_deri=np.array([[]]),
                                  d_locs_coords=np.array([[]]), mu=np.array([[]])):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < chunk_size:
        d_val[pos] = (math.exp(1) ** (
                    -(((d_locs_coords[d_idx_i[pos + start_idx], 0] - mu[d_locs_time[d_idx_i[pos + start_idx]], 0]) -
                       (d_locs_coords[d_idx_j[pos + start_idx], 0] - mu[
                           d_locs_time[d_idx_j[pos + start_idx]], 0])) ** 2 +
                      ((d_locs_coords[d_idx_i[pos + start_idx], 1] - mu[d_locs_time[d_idx_i[pos + start_idx]], 1]) -
                       (d_locs_coords[d_idx_j[pos + start_idx], 1] - mu[
                           d_locs_time[d_idx_j[pos + start_idx]], 1])) ** 2) /
                    ((d_sigma * d_sigma_factor) ** 2)))
        cuda.atomic.add(d_deri, (d_locs_time[d_idx_i[pos + start_idx]], 0),
                        d_val[pos] * 2. * (d_locs_coords[d_idx_i[pos + start_idx], 0] - d_locs_coords[
                            d_idx_j[pos + start_idx], 0] +
                                           mu[d_locs_time[d_idx_j[pos + start_idx]], 0] - mu[
                                               d_locs_time[d_idx_i[pos + start_idx]], 0])
                        / (d_sigma * d_sigma_factor) ** 2)
        cuda.atomic.add(d_deri, (d_locs_time[d_idx_i[pos + start_idx]], 1), d_val[pos] * 2. * (
                d_locs_coords[d_idx_i[pos + start_idx], 1] - d_locs_coords[d_idx_j[pos + start_idx], 1] + mu[
            d_locs_time[d_idx_j[pos + start_idx]], 1] -
                mu[d_locs_time[d_idx_i[pos + start_idx]], 1]) / (d_sigma * d_sigma_factor) ** 2)
        cuda.atomic.add(d_val_sum, 0, d_val[pos])
        d_val[pos] = 0


def cuda_wrapper_chunked(mu, d_locs_coords, d_locs_time, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val, d_deri,
                         chunk_size):
    val = 0
    d_val_sum = cuda.to_device(np.zeros(1, dtype=np.float64))
    #mu = np.append([0, 0], np.asarray(mu))
    mu = np.asarray(mu.reshape(int(mu.size / 2), 2), dtype=np.float64)
    n_chunks = int(np.ceil(d_idx_i.size / chunk_size))
    for i in range(n_chunks - 1):
        threadsperblock = 256
        idc = np.arange(chunk_size) + i * chunk_size
        blockspergrid = (d_idx_j.size + (threadsperblock - 1)) // threadsperblock
        cost_function_full_2d_chunked[blockspergrid, threadsperblock](d_locs_time, idc[0], chunk_size, d_idx_i, d_idx_j,
                                                                      d_sigma,
                                                                      d_sigma_factor, d_val, d_val_sum,
                                                                      d_deri, d_locs_coords, mu)
        val += d_val_sum.copy_to_host()
    n_remaining = d_idx_i.size - (n_chunks - 1) * chunk_size
    idc = np.arange(n_remaining) + (n_chunks - 1) * chunk_size
    threadsperblock = 256
    blockspergrid = (n_remaining + (threadsperblock - 1)) // threadsperblock
    cost_function_full_2d_chunked[blockspergrid, threadsperblock](d_locs_time, idc[0], n_remaining, d_idx_i, d_idx_j,
                                                                  d_sigma, d_sigma_factor, d_val, d_val_sum, d_deri,
                                                                  d_locs_coords, mu)
    val += d_val_sum.copy_to_host()[:n_remaining]
    deri = d_deri.copy_to_host()
    d_deri[:] = 0
    return -np.nansum(val), -deri.flatten()


def optimize_2d_chunked(n_segments, locs_nm, sigma_nm=30, drift_max_nm=300, sigma_factor=1, threshold_estimator_nm=5,
                        display_steps=False):
    drift_estimate = np.zeros(2 * n_segments)
    bounds = []
    for i in range(n_segments):
        bounds.append(((i + 1) * -drift_max_nm, (i + 1) * drift_max_nm))
        bounds.append(((i + 1) * -drift_max_nm, (i + 1) * drift_max_nm))

    idx_i, idx_j = pair_indices_lex_floor(locs_nm[:, :2].copy(), drift_max_nm)

    d_locs_coords = cuda.to_device(np.asarray(locs_nm[:, :2], dtype=np.float32).copy())
    d_locs_time = cuda.to_device(np.asarray(locs_nm[:, 2].astype(int), dtype=np.int32).copy())
    idx_i = np.asarray(np.concatenate(idx_i, dtype=np.int32).ravel())
    idx_j = np.asarray(np.concatenate(idx_j, dtype=np.int32).ravel())
    if len(idx_i) * 32 / 8 > 1000000000:  # array size in the GB range -> use mapped memory
        d_idx_i = cuda.mapped_array_like(idx_i, wc=True)
        d_idx_j = cuda.mapped_array_like(idx_j, wc=True)
        d_idx_i[:] = idx_i
        d_idx_j[:] = idx_j
    else:
        d_idx_i = cuda.to_device(idx_i)
        d_idx_j = cuda.to_device(idx_j)
    d_sigma = np.float64(sigma_nm)
    chunk_size = int(1E7)
    d_val = cuda.to_device(np.zeros(chunk_size))
    deri = np.zeros((n_segments, 2), dtype=np.float64)
    d_deri = cuda.to_device(deri)

    optimization_done = 0
    fails = 0
    while optimization_done == 0:
        d_sigma_factor = np.float64(sigma_factor)
        result = minimize(cuda_wrapper_chunked, drift_estimate, method='L-BFGS-B', options={'disp': display_steps},
                          bounds=bounds, jac=True,
                          args=(d_locs_coords, d_locs_time, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val, d_deri,
                                chunk_size))
        if np.mean(np.abs(result['x'] - drift_estimate)) < threshold_estimator_nm and result['success']:
            optimization_done = 1
        elif not result['success']:
            fails += 1
            if fails > 2:
                sigma_factor = sigma_factor * 2
            elif fails > 5:
                raise RuntimeError('L-BFGS-B Optimization failed')
        else:
            sigma_factor = sigma_factor / 2
        drift_estimate = result['x']
    return drift_estimate


@cuda.jit
def cost_function_full_2d(d_locs_time, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val, d_deri=np.array([[]]),
                          d_locs_coords=np.array([[]]), mu=np.array([[]])):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < d_idx_i.size:
        d_val[pos] = (math.exp(1) ** (-(((d_locs_coords[d_idx_i[pos], 0] - mu[d_locs_time[d_idx_i[pos]], 0]) -
                                         (d_locs_coords[d_idx_j[pos], 0] - mu[d_locs_time[d_idx_j[pos]], 0])) ** 2 +
                                        ((d_locs_coords[d_idx_i[pos], 1] - mu[d_locs_time[d_idx_i[pos]], 1]) -
                                         (d_locs_coords[d_idx_j[pos], 1] - mu[d_locs_time[d_idx_j[pos]], 1])) ** 2) /
                                      ((d_sigma * d_sigma_factor) ** 2)))
        cuda.atomic.add(d_deri, (d_locs_time[d_idx_i[pos]], 0),
                        d_val[pos] * 2. * (d_locs_coords[d_idx_i[pos], 0] - d_locs_coords[d_idx_j[pos], 0] +
                                           mu[d_locs_time[d_idx_j[pos]], 0] - mu[d_locs_time[d_idx_i[pos]], 0])
                        / (d_sigma * d_sigma_factor) ** 2)
        cuda.atomic.add(d_deri, (d_locs_time[d_idx_i[pos]], 1), d_val[pos] * 2. * (
                d_locs_coords[d_idx_i[pos], 1] - d_locs_coords[d_idx_j[pos], 1] + mu[d_locs_time[d_idx_j[pos]], 1] -
                mu[d_locs_time[d_idx_i[pos]], 1]) / (d_sigma * d_sigma_factor) ** 2)


def cuda_wrapper(mu, d_locs_coords, d_locs_time, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val, d_deri):
    mu = np.append([0, 0], np.asarray(mu))
    mu = np.asarray(mu.reshape(int(mu.size / 2), 2), dtype=np.float64)
    threadsperblock = 256
    blockspergrid = (d_idx_j.size + (threadsperblock - 1)) // threadsperblock
    cost_function_full_2d[blockspergrid, threadsperblock](d_locs_time, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val,
                                                          d_deri, d_locs_coords, mu)
    # cuda.synchronize()
    deri = d_deri.copy_to_host()
    d_deri[:] = 0
    return -np.nansum(d_val), -deri.flatten()[2:]


def optimize_2d(n_segments, locs_nm, sigma_nm=30, drift_max_nm=300, sigma_factor=1, threshold_estimator_nm=5,
                display_steps=False):
    drift_estimate = np.zeros(2 * n_segments)
    bounds = []
    for i in range(n_segments - 1):
        bounds.append(((i + 1) * -drift_max_nm, (i + 1) * drift_max_nm))
        bounds.append(((i + 1) * -drift_max_nm, (i + 1) * drift_max_nm))

    idx_i, idx_j = pair_indices_lex(locs_nm[:, :2].copy(), drift_max_nm)

    d_locs_coords = cuda.to_device(np.asarray(locs_nm[:, :2], dtype=np.float64).copy())
    d_locs_time = cuda.to_device(np.asarray(locs_nm[:, 2].astype(int), dtype=np.int64).copy())
    d_idx_i = cuda.to_device(np.asarray(np.concatenate(idx_i).ravel()))
    d_idx_j = cuda.to_device(np.asarray(np.concatenate(idx_j).ravel()))
    d_sigma = np.float64(sigma_nm)
    val = np.zeros_like(d_idx_i, np.float64)
    d_val = cuda.to_device(val)
    deri = np.zeros((n_segments, 2), dtype=np.float64)
    d_deri = cuda.to_device(deri)

    optimization_done = 0
    fails = 0
    while optimization_done == 0:
        d_sigma_factor = np.float64(sigma_factor)
        result = minimize(cuda_wrapper, drift_estimate[2:], method='L-BFGS-B', options={'disp': display_steps},
                          bounds=bounds, jac=True,
                          args=(d_locs_coords, d_locs_time, d_idx_i, d_idx_j, d_sigma, d_sigma_factor, d_val, d_deri))
        if np.mean(np.abs(result['x'] - drift_estimate[2:])) < threshold_estimator_nm and result['success']:
            optimization_done = 1
        elif not result['success']:
            fails += 1
            if fails > 2:
                sigma_factor = sigma_factor * 2
            elif fails > 5:
                raise RuntimeError('L-BFGS-B Optimization failed')
        else:
            sigma_factor = sigma_factor / 2
        drift_estimate[2:] = result['x']
    return drift_estimate
