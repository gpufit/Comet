from tkinter.filedialog import asksaveasfilename
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from scipy.ndimage import convolve
from scipy.optimize import minimize
from comet.core.cuda_wrapper import cuda_wrapper_chunked
from comet.core.cuda_wrapper.cuda_wrapper_qc import cuda_wrapper_chunked_qc
from comet.core.pair_indices import pair_indices_kdtree

from comet.core.pair_indices import estimate_pairs

try:
    from comet.core.pytorch_wrapper.pytorch_util import device_available
    from comet.core.pytorch_wrapper.pytorch_wrapper import torch_wrapper_chunked
    from comet.core.qc_utils import plot_q_with_baseline
    from comet.core.pytorch_wrapper.pytorch_wrapper_qc import torch_wrapper_chunked_qc
except ModuleNotFoundError:
    pass # torch not installed
from comet.core.segmenter import segmentation_wrapper
from comet.core.cpu_wrapper import cpu_wrapper_chunked
from comet.core.interpolation import interpolate_drift
import time

from comet.core.io_utils import save_dataset_as_ms_h5, save_drift_correction_details


def comet_run_kd(dataset, segmentation_mode, segmentation_var, max_locs_per_segment=None,
                 initial_sigma_nm=None, gt_drift=None, display=False, return_corrected_locs=False,
                 max_drift_nm=300, target_sigma_nm=1, boxcar_width=1, drift_max_bound_factor=2,
                 save_corrected_locs=False, save_filepath=None, save_intermediate_results=False,
                 save_correction_details=False,
                 interpolation_method='cubic', mode="cuda", min_max_frames=None,
                 pair_indices_safety_check=False):
    """
        Run COMET drift correction end-to-end.

        Pipeline: temporal segmentation -> KD-tree neighbor pairs -> cost optimization (L-BFGS-B)
        -> optional temporal smoothing -> spline interpolation to per-frame drift -> (optional) subtract drift.

        Parameters
        ----------
        dataset : ndarray of shape (N, 4)
            Localization array with columns [x_nm, y_nm, z_nm, frame]. Units in nm; frame is int.
            For 2D CSVs, insert a zero z column to get (N, 4).
        segmentation_mode : {0, 1, 2}
            Temporal segmentation mode:
            0 = number of windows (choose S directly),
            1 = localizations per window (accumulate frames until >= X locs),
            2 = fixed frame window size (default).
        segmentation_var : int
            Mode-dependent value (S, locs per window, or frames per window).
        initial_sigma_nm : float, default=100
            Initial Gaussian length scale for the overlap kernel (coarse scale).
        target_sigma_nm : float, default=1
            Target (final) Gaussian length scale for fine refinement.
        max_drift_nm : float or None, default=None
            Pair radius in nm used for neighbor search. If None, uses 3 * initial_sigma_nm.
        drift_max_bound_factor : float, default=1.0
            Multiplicative factor for L-BFGS-B box bounds around +-max_drift.
        boxcar_width : int, default=1
            Temporal smoothing width (segments) applied to the estimated drift between optimizer steps.
        interpolation_method : {"cubic", "catmull-rom"}, default="cubic"
            Spline used to convert per-segment drift to per-frame drift.
        max_locs_per_segment : int or None, default=None
            Optional downsampling cap per segment (to control memory/time).
        mode : str, default=cuda, options={"cuda", "cpu", "torch"}
            If True, bypass CUDA path and use CPU backend.
        return_corrected_locs : bool, default=False
            If True, also return drift-corrected localizations.

        Returns
        -------
        drift_interp_with_frames : ndarray of shape (F, 4)
            Per-frame drift with columns [dx_nm, dy_nm, dz_nm, frame].
        corrected_locs : ndarray of shape (N, 4), optional
            Only if return_corrected_locs=True. Columns are [x_nm, y_nm, z_nm, segment_id].
        """

    loc_frames = dataset[:, -1]
    if min_max_frames is None:
        min_max_frames = (loc_frames.min(), loc_frames.max())

    # Segment the dataset based on frame numbers into time windows

    result, sorted_dataset, idx_i, idx_j = segmentation_and_pair_indices_wrapper(
        dataset, segmentation_var, segmentation_mode, max_drift_nm, max_locs_per_segment,
    pair_indices_safety_check=pair_indices_safety_check)

    # Set default initial sigma if not provided
    if initial_sigma_nm is None:
        initial_sigma_nm = max_drift_nm // 3

    # Run drift optimization
    t0 = time.time()
    drift_est = optimize_3d_chunked_better_moving_avg_kd(
        result.n_segments, sorted_dataset,
        idx_i, idx_j,
        sigma_nm=initial_sigma_nm,
        target_sigma_nm=target_sigma_nm,
        drift_max_nm=max_drift_nm,
        drift_max_bound_factor=drift_max_bound_factor,
        display_steps=display,
        boxcar_width=boxcar_width,
        segmentation_result=result,
        save_intermdiate_results=save_intermediate_results,
        mode=mode
    )
    elapsed = time.time() - t0

    # Reshape and interpolate drift across all frames
    drift_est = drift_est.reshape((result.n_segments, 3))
    vld_tp = np.where(~np.isnan(drift_est[:, 0]))

    frame_interp = np.arange(0, min_max_frames[1] + 1, dtype=int)
    drift_interp = interpolate_drift(result.center_frames[vld_tp], drift_est[vld_tp], frame_interp,
                                     method=interpolation_method)
    drift_interp_with_frames = np.hstack((drift_interp, frame_interp[:, np.newaxis]))

    # Apply drift correction to original localizations
    for i in range(3):
        dataset[:, i] = dataset[:, i] - drift_interp[dataset[:, -1].astype(int), i]

    # Optionally show estimated drift curve
    if display:
        print(f"Drift estimation completed in {elapsed:.2f} seconds.")
        plt.figure()
        plt.plot(frame_interp, drift_interp)
        plt.title("Estimated Drift")
        plt.xlabel("Frames")
        plt.ylabel("Drift (nm)")
        plt.legend(['X', 'Y', 'Z'])
        plt.show()


    # Optional GT comparison plot
    if display and gt_drift is not None:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(gt_drift[:, 3], gt_drift[:, 0], label='GT Drift X')
        ax[1].plot(gt_drift[:, 3], gt_drift[:, 1], label='GT Drift Y')
        ax[2].plot(gt_drift[:, 3], gt_drift[:, 2], label='GT Drift Z')
        ax[0].plot(frame_interp, drift_interp[:, 0], label='Estimated Drift X', linestyle='--')
        ax[1].plot(frame_interp, drift_interp[:, 1], label='Estimated Drift Y', linestyle='--')
        ax[2].plot(frame_interp, drift_interp[:, 2], label='Estimated Drift Z', linestyle='--')
        ax[1].set_title("Ground Truth vs Estimated Drift")
        ax[1].set_xlabel("Frames")
        ax[0].set_ylabel("Drift (nm)")
        plt.legend()
        plt.show()

    # Optional: Save corrected localizations
    if save_corrected_locs:
        if save_filepath is None:
            save_filepath = asksaveasfilename(title="Save drift corrected localizations as molecule set",
                                              defaultextension='h5')
        save_dataset_as_ms_h5(sorted_dataset[:, :-1], sorted_dataset[:, -1], 160, filename=save_filepath)

    if save_correction_details:
        if save_filepath is None:
            save_filepath = asksaveasfilename(title="Save drift correction details in h5 file",
                                              defaultextension='h5')
        else:
            save_filepath = save_filepath.replace(".h5", "_details.h5")

        save_drift_correction_details(save_filepath, drift_est, drift_interp, frame_interp,
                                      result, elapsed, initial_sigma_nm, target_sigma_nm, gt_drift=None)


    # Return corrected locs + drift
    if return_corrected_locs:
        return drift_interp_with_frames, dataset
    else:
        return drift_interp_with_frames


def optimize_3d_chunked_better_moving_avg_kd(n_segments, locs_nm, idx_i, idx_j, sigma_nm=30, drift_max_nm=300,
                                             target_sigma_nm=30, display_steps=False,
                                             save_intermdiate_results=False,
                                             boxcar_width=3, drift_max_bound_factor=2,
                                             segmentation_result=None,
                                             mode="cuda", return_calc_time=False):
    """
    Estimate per-segment drift (mu) by minimizing the negative Gaussian-overlap cost
    with an L-BFGS-B optimizer and a coarse-to-fine schedule on sigma.

    The routine operates on temporally segmented localizations and reuses a static
    neighbor graph (pairs within drift_max_nm). Between optimizer steps, a moving
    average (boxcar) can be applied to mu as temporal regularization. Sigma is
    reduced iteratively from `sigma_nm` toward `target_sigma_nm` for robust
    convergence.

    Parameters
    ----------
    n_segments : int
        Number of temporal segments S (0..S-1). One 3D drift vector is estimated per segment.
    locs_nm : ndarray of shape (M, 3)
        Kept localizations in nanometers, columns [x, y, z]; typically already masked/downsampled.
    sigma_nm : float, default=30
        Initial Gaussian width (coarse scale) for the overlap kernel.
    drift_max_nm : float, default=300
        Maximum expected drift (nm). Also used as the radius for neighbor pairs and
        as the L-BFGS-B bound scale (see `drift_max_bound_factor`).
    target_sigma_nm : float, default=30
        Target / final Gaussian width (fine scale). The optimizer reduces sigma toward
        this value over iterations.
    display_steps : bool, default=False
        If True, print or log intermediate progress per iteration/scale.
    save_intermdiate_results : bool, default=False
        If True, save intermediate states (e.g., mu after iterations/scales) for inspection.
        (Name kept as in code.)
    boxcar_width : int, default=3
        Temporal smoothing width (in segments) for a moving average applied to mu between steps.
        Use 0 or 1 to disable smoothing.
    drift_max_bound_factor : float, default=2
        Multiplier for L-BFGS-B bounds around +- drift_max_nm to keep updates physically reasonable.
    segmentation_result : object or None, default=None
        Segmentation info and metadata. Expected to provide, at minimum:
        - segment IDs per localization
        - center frames per segment
        - any additional structures required by backend (e.g., pair indices)
        If None, pairs/ids may be built internally depending on implementation.
    mode : str, default=cuda
        If True, use the CPU backend; otherwise try GPU (CUDA) and fall back to CPU if unavailable.
    return_calc_time : bool, default=False
        If True, also return the total computation time in seconds.

    Returns
    -------
    mu : ndarray of shape (S, 3)
        Estimated per-segment drift (dx, dy, dz) in nanometers.
    calc_time_s : float, optional
        Only when `return_calc_time=True`. Wall-clock time for the optimization.
        """

    if segmentation_result is None:
        segmentation_result = {}
    intermediate_results_filehandle = None
    sigma_factor = 1.0

    # Extract coordinate + time arrays, convert to device if CUDA
    coords = locs_nm[:, :3].astype(np.float32).copy()
    times = locs_nm[:, 3].astype(np.int32).copy()

    chunk_size = int(1E8)  # 1E7

    quality_control = mode == "torch_qc" or mode == "cuda_qc"

    if mode == "cuda" or mode == "cuda_qc":
        d_coords = cuda.to_device(coords)
        d_times = cuda.to_device(times)
        if len(idx_i) * 4 > 2e9:
            # Use mapped memory if index arrays are large
            print("Large index arrays — using mapped memory.")
            d_idx_i = cuda.mapped_array_like(idx_i.astype(np.int32), wc=True)
            d_idx_j = cuda.mapped_array_like(idx_j.astype(np.int32), wc=True)
            d_idx_i[:] = idx_i
            d_idx_j[:] = idx_j
        else:
            d_idx_i = cuda.to_device(idx_i.astype(np.int32))
            d_idx_j = cuda.to_device(idx_j.astype(np.int32))
        # Preallocate device arrays
        d_sigma = np.float64(sigma_nm)
        d_val = cuda.to_device(np.zeros(chunk_size))
        d_deri = cuda.to_device(np.zeros((n_segments, 3), dtype=np.float64))
        wrapper = cuda_wrapper_chunked
        if quality_control:
            qc_wrapper = cuda_wrapper_chunked_qc
    elif mode == "torch" or mode == "torch_qc":
        try:
            import torch
            device = device_available()
        except ImportError:
            raise ImportError("Torch is not installed or no suitable device found for torch mode.")
        d_coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        d_times = torch.as_tensor(times, dtype=torch.int64, device=device)
        d_idx_i = torch.as_tensor(idx_i, dtype=torch.int64, device=device)
        d_idx_j = torch.as_tensor(idx_j, dtype=torch.int64, device=device)
        d_sigma = torch.as_tensor(sigma_nm, dtype=torch.float32, device=device)
        d_val = device  # not used for torch implementation, for now abused to pass device
        d_deri = None  # not needed for torch implementation
        wrapper = torch_wrapper_chunked
        if quality_control:
            qc_wrapper = torch_wrapper_chunked_qc
    else:
        # Fallback: CPU arrays
        print("Warning: Using CPU Backend.")
        d_coords = coords
        d_times = times
        d_sigma = sigma_nm
        d_idx_i, d_idx_j = idx_i, idx_j
        d_val = np.zeros(len(idx_i), dtype=np.float64)
        d_deri = np.zeros((n_segments, 3), dtype=np.float64)
        wrapper = cpu_wrapper_chunked
        if quality_control:
            qc_wrapper = None
            raise RuntimeError("Quality control mode not implemented for CPU backend.")

    # Initial drift estimate + bounds
    drift_est = np.zeros(n_segments * 3)
    bounds = [(-drift_max_nm * drift_max_bound_factor, drift_max_nm * drift_max_bound_factor)] * (3 * n_segments)

    drift_est_gradient = np.inf
    fails = 0
    done = False
    itr_counter = 0
    start_time = time.time()

    while not done:
        # Apply boxcar smoothing to current estimate
        tmp = drift_est.reshape((-1, 3))
        for i in range(3):
            tmp[:, i] = convolve(tmp[:, i], np.ones(boxcar_width) / boxcar_width)
        drift_est = tmp.flatten()

        # Run L-BFGS-B optimization step
        result = minimize(wrapper, drift_est, method='L-BFGS-B',
                          args=(
                              d_coords, d_times, d_idx_i, d_idx_j, d_sigma, sigma_factor, d_val, d_deri, chunk_size),
                          jac=True, bounds=bounds,
                          options={'disp': display_steps, 'gtol': 1E-5, 'ftol': 1E3 * np.finfo(float).eps,
                                   'maxls': 40})
        itr_counter += 1
        print(f"Iteration {itr_counter}: status = {result.status}, success = {result.success}")
        print(f"  current sigma: {np.round(sigma_nm * sigma_factor, 2)} nm")

        # Optionally save intermediate result to HDF5
        if save_intermdiate_results:
            intermediate_results_filehandle = save_intermediate_results_wrapper(drift_est_nm=drift_est.reshape(-1, 3),
                                                                                locs_nm=locs_nm,
                                                                                sigma_nm=sigma_nm,
                                                                                sigma_factor=sigma_factor,
                                                                                itr_counter=itr_counter, fails=fails,
                                                                                segmentation_result=segmentation_result,
                                                                                filehandle=intermediate_results_filehandle)
        # Update if successful
        if result.success:
            delta = np.median((result.x - drift_est) ** 2)
            print(f"  drift estimate gradient: {delta}")
            print(f"  previous gradient: {drift_est_gradient}")
            # Check convergence
            if (delta > drift_est_gradient or sigma_nm * sigma_factor <= 1.0) and sigma_nm * sigma_factor <= target_sigma_nm:
                done = True
                calc_time = time.time() - start_time
                print(f"Optimization completed in {calc_time:.2f} s")
            else:
                sigma_factor /= 1.5
                drift_est_gradient = delta
                drift_est = result.x
        else:
            fails += 1
            if fails > 2:
                sigma_factor *= 2
                print("Restarting with larger sigma_factor")
            if fails > 5:
                raise RuntimeError("L-BFGS-B Optimization failed after multiple retries")

    if quality_control:
        # Experimental feature: Still under review!
        # Compute and plot quality metrics after each successful optimization step
        loss, grad, qc = qc_wrapper(
            drift_est,
            d_coords, d_times,
            d_idx_i, d_idx_j,
            d_sigma, sigma_factor,
            d_val, d_deri,
            chunk_size
        )
        idx_flawed = plot_q_with_baseline(qc["Q_obs"], qc["Q_null"],
        pairs=qc["window_pairs"],
        window=qc["window"],
        title=f"Normalized overlap per pair ({mode})")
        if len(idx_flawed) > 0:
            plt.figure()
            plt.plot(drift_est.reshape((-1, 3)))
            plt.vlines(idx_flawed, np.min(drift_est), np.max(drift_est), color='r', label='pot. flawed indices', alpha=0.4)
            plt.legend()
        plt.show()
        drift_est = drift_est.reshape((-1, 3))
        drift_est[idx_flawed] = np.nan
        drift_est = drift_est.flatten()
        ###################
    if return_calc_time:
        return drift_est, time.time() - start_time, itr_counter
    else:
        return drift_est


def segmentation_and_pair_indices_wrapper(dataset, segmentation_var, segmentation_mode, max_drift_nm,
                                          max_locs_per_segment, pair_indices_safety_check=False, hard_limit_pairs=None):
    if not segmentation_mode == -1: # -1 is for pre-segmented data
        result = segmentation_wrapper(dataset[:, -1], segmentation_var, segmentation_mode,
                                      max_locs_per_segment, return_param_dict=True)
    else:
        # pre segmented data, anyway we set these values in case auto downsampling is needed
        segmentation_mode = 2  # dummy --> segment per frame ...
        segmentation_var = 1    # dummy --> ... using 1 frame per segment
    if pair_indices_safety_check:
        n_pairs_est = estimate_pairs(dataset[result.loc_valid, :3], max_drift_nm)
        print(f"Estimated number of pairs within {max_drift_nm} nm: {n_pairs_est:,}")
        if hard_limit_pairs is not None and n_pairs_est > hard_limit_pairs:
            raise RuntimeError(f"Estimated number of pairs {n_pairs_est} exceeds hard limit of {hard_limit_pairs}. "
                               f"Aborting to avoid crash.")
        if n_pairs_est > 5e8:
            print(f"Estimated number of pairs is very large {n_pairs_est}. "
                  f"Automatic down-sampling is usually required above 500 mil."
                  f"Billions of pairs can lead to crash.")
            ans = input("Continue anyway? (y/n): ")
            if ans.lower() != 'y':
                raise RuntimeError("Aborted by user due to large estimated number of pairs.")
    idx_i, idx_j, successful = pair_indices_kdtree(dataset[result.loc_valid, :3], max_drift_nm)
    if not successful:
        if max_locs_per_segment is None:
            max_locs_per_segment = int(result.out_dict['locs_per_segment'].max())
    while not successful:
        max_locs_per_segment = int(max_locs_per_segment * 0.9)
        print(f"Segmentation and Pairing attempt failed, automatic down-sampling active...")
        print(f"Retrying segmentation with max_locs_per_segment={max_locs_per_segment}...")
        result = segmentation_wrapper(dataset[:, -1], segmentation_var, segmentation_mode,
                                      max_locs_per_segment, return_param_dict=True)
        sorted_dataset = dataset.copy()
        sorted_dataset[:, -1] = result.loc_segments
        sorted_dataset = sorted_dataset[result.loc_valid]
        idx_i, idx_j, successful = pair_indices_kdtree(sorted_dataset[:, :3], max_drift_nm)
    print(f"Segmentation and Pairing successful resulting in {result.n_segments:,} time windows with on average "
          f"{int(np.median(result.out_dict['locs_per_segment']))} locs per time window. "
          f"{len(idx_i):,} Pairs where found.")
    sorted_dataset = dataset.copy()
    sorted_dataset[:, -1] = result.loc_segments
    sorted_dataset = sorted_dataset[result.loc_valid]
    return result, sorted_dataset, idx_i, idx_j


def save_intermediate_results_wrapper(drift_est_nm, locs_nm, sigma_nm, sigma_factor, itr_counter, fails,
                                      segmentation_result, filehandle=None):
    if filehandle is None:
        filename = asksaveasfilename(title="Save intermediate results h5 file",
                                     defaultextension=".h5")
        filehandle = h5py.File(filename, 'a')
    # Save drift + state info under a named HDF5 group
    filehandle.create_group(f"iteration_{itr_counter}")
    filehandle[f"iteration_{itr_counter}"]['drift_estimate_nm'] = drift_est_nm
    filehandle[f"iteration_{itr_counter}"]['locs_nm'] = locs_nm
    filehandle[f"iteration_{itr_counter}"]['sigma_nm'] = sigma_nm
    filehandle[f"iteration_{itr_counter}"]['sigma_factor'] = sigma_factor
    filehandle[f"iteration_{itr_counter}"]['fails'] = fails
    filehandle[f"iteration_{itr_counter}"]['timestamp'] = time.time()
    if "segmentation_result" not in filehandle.keys():
        filehandle.create_group("segmentation_result")
        filehandle["segmentation_result"]["n_segments"] = segmentation_result.n_segments
        filehandle["segmentation_result"]["center_frames"] = segmentation_result.center_frames
        filehandle["segmentation_result"]["loc_segments"] = segmentation_result.loc_segments
        filehandle["segmentation_result"]["loc_valid"] = segmentation_result.loc_valid
    return filehandle
