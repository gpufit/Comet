import math
import numpy as np
import matplotlib.pyplot as plt


def cost_function_full_3d_chunked_cpu(
    locs_time, start_idx, chunk_size,
    idx_i, idx_j,
    sigma, sigma_factor,
    val, val_sum,
    deri, locs_coords, mu
):
    sigma_sq = (2 * sigma * sigma_factor) ** 2

    for pos in range(chunk_size):
        i = idx_i[pos + start_idx]
        j = idx_j[pos + start_idx]

        ti = locs_time[i]
        tj = locs_time[j]

        dx = (locs_coords[i, 0] - mu[ti, 0]) - (locs_coords[j, 0] - mu[tj, 0])
        dy = (locs_coords[i, 1] - mu[ti, 1]) - (locs_coords[j, 1] - mu[tj, 1])
        dz = (locs_coords[i, 2] - mu[ti, 2]) - (locs_coords[j, 2] - mu[tj, 2])

        diff_sq = dx * dx + dy * dy + dz * dz
        val[pos] = math.exp(-diff_sq / sigma_sq) / (sigma * sigma_factor)

        # Derivative contributions — match GPU kernel
        for dim in range(3):
            deri[tj, dim] += 2 * val[pos] * (
                locs_coords[j, dim] - locs_coords[i, dim] +
                mu[ti, dim] - mu[tj, dim]
            ) / sigma_sq

            deri[ti, dim] += 2 * val[pos] * (
                locs_coords[i, dim] - locs_coords[j, dim] +
                mu[tj, dim] - mu[ti, dim]
            ) / sigma_sq

        val_sum += val[pos]
        val[pos] = 0

    return val_sum, deri


def cuda_wrapper_chunked_cpu(
    mu, locs_coords, locs_time,
    idx_i, idx_j,
    sigma, sigma_factor,
    val, deri, chunk_size,
    debug=False
):
    val_total = 0.0
    mu = mu.reshape((-1, 3)).astype(np.float64)
    n_pairs = len(idx_i)
    n_chunks = int(np.ceil(n_pairs / chunk_size))

    for i in range(n_chunks):
        start_idx = i * chunk_size
        current_chunk = min(chunk_size, n_pairs - start_idx)

        val_sum = 0.0  # scalar accumulator
        val_sum, deri = cost_function_full_3d_chunked_cpu(
            locs_time, start_idx, current_chunk,
            idx_i, idx_j,
            sigma, sigma_factor,
            val, val_sum,
            deri, locs_coords, mu
        )

        val_total += val_sum

    gradient = deri.copy()
    deri[:] = 0  # zero for reuse

    if debug:
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].plot(gradient[:, 0])
        ax[1, 0].plot(gradient[:, 1])
        ax[2, 0].plot(gradient[:, 2])
        ax[0, 0].set_title(f"Gradients (sigma={sigma * sigma_factor:.2f} nm)")

        ax[0, 1].plot(mu[:, 0])
        ax[1, 1].plot(mu[:, 1])
        ax[2, 1].plot(mu[:, 2])
        ax[0, 1].set_title("Drift Estimate [nm]")
        plt.tight_layout()
        plt.show()

    return -val_total, -gradient.flatten()

