import numpy as np
import matplotlib.pyplot as plt
from comet.core.drift_optimizer import comet_run_kd
import time


def generate_synthetic_dataset(n_frames=40, n_locs_per_frame=50, n_gt_coords=20, spacing=5000):
    """
    Generates synthetic data:
    - fixed localizations per fram
    - repeated across n frames
    - linear + sine drift in X, mild linear in Y
    """
    frames = np.repeat(np.arange(n_frames), n_locs_per_frame)

    # Create fixed locs (spread out to limit pair count)
    base_coords = np.random.rand(n_gt_coords, 3) * spacing

    coords = []
    for f in range(n_frames):
        chosen_indices = np.random.choice(n_gt_coords, n_locs_per_frame, replace=True)
        coords.append(base_coords[chosen_indices] + np.random.normal(0, 3, (n_locs_per_frame, 3)))
    coords = np.vstack(coords)

    # Create known drift
    gt_drift = np.zeros((n_frames, 3))
    gt_drift[:, 0] = 5 * np.sin(np.linspace(0, 4 * np.pi, n_frames)) + np.linspace(0, 20, n_frames)
    gt_drift[:, 1] = np.linspace(0, 5, n_frames)
    gt_drift[:, 2] = 0  # No Z drift

    coords[:, 0] += np.repeat(gt_drift[:, 0], n_locs_per_frame)
    coords[:, 1] += np.repeat(gt_drift[:, 1], n_locs_per_frame)

    locs = np.column_stack([coords, frames])
    return locs, gt_drift


def test_cpu_vs_cuda_on_synthetic(plot=False):
    locs, gt_drift = generate_synthetic_dataset()

    cpu_timer = time.time()
    drift_cpu, _ = comet_run_kd(
        dataset=locs.copy(),
        segmentation_mode=2,
        segmentation_var=1,
        initial_sigma_nm=100,
        max_drift_nm=100,
        target_sigma_nm=10,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        display=True,
        mode="cpu"
    )
    cpu_time = time.time() - cpu_timer
    cuda_timer = time.time()
    drift_cuda, _ = comet_run_kd(
        dataset=locs.copy(),
        segmentation_mode=2,
        segmentation_var=1,
        initial_sigma_nm=100,
        max_drift_nm=100,
        target_sigma_nm=10,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        display=True,
        mode="cuda"
    )
    cuda_time = time.time() - cuda_timer
    torch_timer = time.time()
    drift_torch, _ = comet_run_kd(
        dataset=locs,
        segmentation_mode=2,
        segmentation_var=1,
        initial_sigma_nm=100,
        max_drift_nm=100,
        target_sigma_nm=10,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        display=True,
        mode="torch"
    )
    torch_time = time.time() - torch_timer

    print(f"CPU time: {cpu_time:.2f}s, CUDA time: {cuda_time:.2f}s, Torch time: {torch_time:.2f}s")

    for i in range(3):
        drift_cuda[:, i] -= np.mean(drift_cuda[:, i])
        drift_cpu[:, i] -= np.mean(drift_cpu[:, i])
        drift_torch[:, i] -= np.mean(drift_torch[:, i])
        gt_drift[:, i] -= np.mean(gt_drift[:, i])

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(drift_cpu[:, 0], label="CPU X")
        plt.plot(drift_cuda[:, 0], "--", label="CUDA X")
        plt.plot(drift_torch[:, 0], "-.", label="Torch X")
        plt.plot(gt_drift[:, 0], ":", label="GT X")
        plt.legend()
        plt.title("Drift Comparison - X Axis")
        plt.xlabel("Frame")
        plt.ylabel("Drift [nm]")
        plt.tight_layout()
        plt.show()

    mae = np.mean(np.abs(drift_cpu - drift_cuda))
    assert mae < 1.0, f"Mean drift difference too large: {mae:.2f} nm"


if __name__ == "__main__":
    test_cpu_vs_cuda_on_synthetic(plot=True)

