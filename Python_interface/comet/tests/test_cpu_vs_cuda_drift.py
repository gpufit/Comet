import numpy as np
import matplotlib.pyplot as plt
from comet.core.drift_optimizer import comet_run_kd


def generate_synthetic_dataset(n_frames=20, n_locs=20, spacing=5000):
    """
    Generates synthetic data:
    - 100 fixed localizations
    - repeated across 100 frames
    - linear + sine drift in X, mild linear in Y
    """
    frames = np.repeat(np.arange(n_frames), n_locs)

    # Create fixed locs (spread out to limit pair count)
    base_coords = np.random.rand(n_locs, 3) * spacing

    # Repeat them across all frames
    coords = np.tile(base_coords, (n_frames, 1))

    # Create known drift
    gt_drift = np.zeros((n_frames, 3))
    gt_drift[:, 0] = 5 * np.sin(np.linspace(0, 4 * np.pi, n_frames)) + np.linspace(0, 20, n_frames)
    gt_drift[:, 1] = np.linspace(0, 5, n_frames)
    gt_drift[:, 2] = 0  # No Z drift

    coords[:, 0] += np.repeat(gt_drift[:, 0], n_locs)
    coords[:, 1] += np.repeat(gt_drift[:, 1], n_locs)

    locs = np.column_stack([coords, frames])
    return locs, gt_drift


def test_cpu_vs_cuda_on_synthetic(plot=False):
    locs, gt_drift = generate_synthetic_dataset()

    drift_cpu, _ = comet_run_kd(
        dataset=locs.copy(),
        segmentation_mode=2,
        segmentation_var=1,
        initial_sigma_nm=100,
        max_drift=100,
        target_sigma_nm=10,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        display=True,
        force_cpu=True
    )

    drift_cuda, _ = comet_run_kd(
        dataset=locs,
        segmentation_mode=2,
        segmentation_var=1,
        initial_sigma_nm=100,
        max_drift=100,
        target_sigma_nm=10,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        display=True,
        force_cpu=False
    )

    for i in range(3):
        drift_cuda[:, i] -= np.mean(drift_cuda[:, i])
        drift_cpu[:, i] -= np.mean(drift_cpu[:, i])
        gt_drift[:, i] -= np.mean(gt_drift[:, i])

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(drift_cpu[:, 0], label="CPU X")
        plt.plot(drift_cuda[:, 0], "--", label="CUDA X")
        plt.plot(gt_drift[:, 0], ":", label="GT X")
        plt.legend()
        plt.title("Drift Comparison - X Axis")
        plt.xlabel("Frame")
        plt.ylabel("Drift [nm]")
        plt.tight_layout()
        plt.show()

    mae = np.mean(np.abs(drift_cpu - drift_cuda))
    assert mae < 1.0, f"Mean drift difference too large: {mae:.2f} nm"

# if __name__ == "__main__":
#     test_cpu_vs_cuda_on_synthetic(plot=True)
