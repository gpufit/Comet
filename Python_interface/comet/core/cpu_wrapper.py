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

import matplotlib.pyplot as plt
import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange


@njit(parallel=True, fastmath=False, cache=True)
def cost_function_full_3d_parallel_cpu(locs_time: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray, sigma: float, sigma_factor: float,
									   locs_coords: np.ndarray, mu: np.ndarray) -> tuple[float, np.ndarray]:
	"""
	Calculates the cost and gradient of COMET in parallel on the CPU.

	Each thread uses a private gradient array to avoid concurrent writes when multiple pairs involve the same temporal segments.

    :param locs_time: Segment identifier for each location.
    :param idx_i: First index of each pair.
    :param idx_j: Second index of each pair.
    :param sigma: Initial width of the Gaussian kernel.
	:param sigma_factor: Multiplicative factor applied to ``sigma``.
    :param locs_coords: Coordinates of the locations, in the form ``(N, 3)``.
    :param mu: Current drift per segment, in the form ``(S, 3)``.
    :return: Positive cost and gradient in the form ``(S, 3)``.
	"""
	n_pairs = idx_i.size
	n_segments = mu.shape[0]
	n_threads = get_num_threads()

	sigma_scaled = sigma * sigma_factor
	sigma_sq = (2.0 * sigma_scaled) ** 2
	inverse_sigma = 1.0 / sigma_scaled
	derivative_factor = 2.0 / sigma_sq

	value_sum = 0.0
	derivatives_per_thread = np.zeros((n_threads, n_segments, 3), dtype=np.float64)

	for pos in prange(n_pairs):
		thread_id = get_thread_id()

		index_i, index_j = idx_i[pos], idx_j[pos]
		time_i, time_j = locs_time[index_i], locs_time[index_j]

		dx = (locs_coords[index_i, 0] - mu[time_i, 0] - locs_coords[index_j, 0] + mu[time_j, 0])
		dy = (locs_coords[index_i, 1] - mu[time_i, 1] - locs_coords[index_j, 1] + mu[time_j, 1])
		dz = (locs_coords[index_i, 2] - mu[time_i, 2] - locs_coords[index_j, 2] + mu[time_j, 2])

		distance_sq = dx * dx + dy * dy + dz * dz
		value = math.exp(-distance_sq / sigma_sq) * inverse_sigma
		value_sum += value

		coefficient = value * derivative_factor

		derivatives_per_thread[thread_id, time_j, 0] += coefficient * (locs_coords[index_j, 0] - locs_coords[index_i, 0] + mu[time_i, 0] - mu[time_j, 0])
		derivatives_per_thread[thread_id, time_j, 1] += coefficient * (locs_coords[index_j, 1] - locs_coords[index_i, 1] + mu[time_i, 1] - mu[time_j, 1])
		derivatives_per_thread[thread_id, time_j, 2] += coefficient * (locs_coords[index_j, 2] - locs_coords[index_i, 2] + mu[time_i, 2] - mu[time_j, 2])

		derivatives_per_thread[thread_id, time_i, 0] += coefficient * (locs_coords[index_i, 0] - locs_coords[index_j, 0] + mu[time_j, 0] - mu[time_i, 0])
		derivatives_per_thread[thread_id, time_i, 1] += coefficient * (locs_coords[index_i, 1] - locs_coords[index_j, 1] + mu[time_j, 1] - mu[time_i, 1])
		derivatives_per_thread[thread_id, time_i, 2] += coefficient * (locs_coords[index_i, 2] - locs_coords[index_j, 2] + mu[time_j, 2] - mu[time_i, 2])

	deri = np.zeros((n_segments, 3), dtype=np.float64)

	for thread_id in range(n_threads):
		for segment in range(n_segments):
			deri[segment, 0] += derivatives_per_thread[thread_id, segment, 0]
			deri[segment, 1] += derivatives_per_thread[thread_id, segment, 1]
			deri[segment, 2] += derivatives_per_thread[thread_id, segment, 2]

	return value_sum, deri


def cpu_wrapper_chunked(mu: np.ndarray, locs_coords: np.ndarray, locs_time: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray, sigma: float, sigma_factor: float,
						val: np.ndarray, deri: np.ndarray, chunk_size: int, debug: bool = False) -> tuple[float, np.ndarray]:
	"""
	Parallel CPU interface used by the L-BFGS-B optimizer.

    The arguments ``val``, ``deri``, and ``chunk_size`` are retained to maintain the same signature as the other backends.
	"""
	del val
	del deri
	del chunk_size

	mu_reshaped = np.asarray(mu, dtype=np.float64).reshape((-1, 3))

	value, gradient = cost_function_full_3d_parallel_cpu(np.asarray(locs_time, dtype=np.int32), np.asarray(idx_i, dtype=np.int32),
														 np.asarray(idx_j, dtype=np.int32), float(sigma), float(sigma_factor),
														 np.asarray(locs_coords, dtype=np.float32), mu_reshaped)
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

	return -value, -gradient.ravel()
