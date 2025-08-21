import os
import numpy as np
from comet.core.io_utils import load_normal_molecule_set, load_simulation_dataset_and_gt_drift
from comet.core.drift_optimizer import comet_run_kd
import matplotlib.pyplot as plt


def data_path(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..\\..", "data", filename))


def test_cuda_on_real_dataset(plot=True):
    h5_file = data_path("npc_n96_3d_prec_2nm_2_loc_per_frm_000.molecule_set.h5")
    dataset = load_normal_molecule_set(h5_file)

    drift_cuda, _ = comet_run_kd(
        dataset=dataset,
        segmentation_mode=2,
        segmentation_var=120,
        initial_sigma_nm=120,
        max_drift=600,
        target_sigma_nm=10,
        drift_max_bound_factor=2,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        force_cpu=False,
        display=True
    )

    if plot:
        plt.figure()
        plt.plot(drift_cuda[:, 0], label="X")
        plt.plot(drift_cuda[:, 1], label="Y")
        plt.plot(drift_cuda[:, 2], label="Z")
        plt.legend()
        plt.title("CUDA Drift Correction Output")
        plt.xlabel("Frame")
        plt.ylabel("Drift [nm]")
        plt.tight_layout()
        plt.show()

    assert drift_cuda.shape[1] == 3
    assert not np.isnan(drift_cuda).any(), "Drift output contains NaNs"


def test_cuda_on_sim_dataset(plot=True):
    h5_file = data_path("mt_2d_data_v3_prec_2_nm_000.h5")
    dataset, gt_drift = load_simulation_dataset_and_gt_drift(h5_file)

    drift_cuda, _ = comet_run_kd(
        dataset=dataset,
        segmentation_mode=2,
        segmentation_var=80,
        initial_sigma_nm=120,
        max_locs_per_segment=250,
        max_drift=400,
        target_sigma_nm=10,
        drift_max_bound_factor=2,
        boxcar_width=3,
        return_corrected_locs=True,
        interpolation_method='cubic',
        force_cpu=False,
        display=True
    )

    if plot:
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        ax[0].plot(drift_cuda[:, 3], drift_cuda[:, 0] - np.mean(drift_cuda[:, 0]), label="COMET")
        ax[1].plot(drift_cuda[:, 3], drift_cuda[:, 1] - np.mean(drift_cuda[:, 1]), label="COMET")
        ax[2].plot(drift_cuda[:, 3], drift_cuda[:, 2] - np.mean(drift_cuda[:, 2]), label="COMET")
        ax[0].plot(gt_drift[:, 0] - np.mean(gt_drift[:, 0]), label="GT", linestyle='--')
        ax[1].plot(gt_drift[:, 1] - np.mean(gt_drift[:, 1]), label="GT", linestyle='--')
        ax[2].plot(gt_drift[:, 2] - np.mean(gt_drift[:, 2]), label="GT", linestyle='--')
        ax[2].legend()
        ax[0].set_title("COMET vs. GT")
        ax[2].set_xlabel("Frame")
        ax[0].set_ylabel("Drift X [nm]")
        ax[1].set_ylabel("Drift Y [nm]")
        ax[2].set_ylabel("Drift Z [nm]")
        plt.tight_layout()
        plt.show()

    assert drift_cuda.shape[1] == 4
    assert not np.isnan(drift_cuda).any(), "Drift output contains NaNs"


if __name__ == "__main__":
    # test_cuda_on_real_dataset(plot=True)
    test_cuda_on_sim_dataset(plot=True)
