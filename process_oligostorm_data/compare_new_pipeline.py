
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Python_interface.comet.core.io_utils import (
        load_normal_molecule_set)
from process_oligostorm_data.oligostorm_comet_pipeline import _render_dataset_2d


def compare_new_pipeline_locs_basis():
    folder = r"\\192.168.1.195\storm_share\storm_disk1\STORM_data1\Optimized_drift_project\Sarah_data\M7_SVABext027\\"
    filename_old = folder + "macro_correction_using_bead_cropped_COMET_on_full_dataset\\SVABext027_MS-rep_loc001_co16_bg50_xy20-z40_COMET_correct_and_aligned_final.h5"
    filename_new = folder + "loc5_correction/SVABext027_MS-rep_loc005_co16_bg50_xy20-z40_ext-thunderstorm.molecule_set.h5"

    dataset_old = load_normal_molecule_set(filename_old)
    dataset_new = load_normal_molecule_set(filename_new)

    for i in range(2):
        dataset_old[:, i] -= np.median(dataset_old[:, i])
        dataset_new[:, i] -= np.median(dataset_new[:, i])

    image_old, extent_old, actual_pixel_size_nm = _render_dataset_2d(
        dataset_old,
        render_sigma_nm=15,
        pixel_size_nm=15,
    )
    image_new, extent_new, actual_pixel_size_nm = _render_dataset_2d(
        dataset_new,
        render_sigma_nm=15,
        pixel_size_nm=15,
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax[0].imshow(np.log(image_old+1), extent=extent_old, cmap='hot', origin='lower')
    ax[0].set_title('Old Pipeline')
    ax[1].imshow(np.log(image_new+1), extent=extent_new, cmap='hot', origin='lower')
    ax[1].set_title('New Pipeline')
    plt.show()


def compare_new_pipeline_drift_curve():

    folder_new = r"\\192.168.1.195\storm_share\storm_disk1\STORM_data1\Optimized_drift_project\Sarah_data\M7_SVABext027\\"
    drift_est_new_file = folder_new + "loc5_correction\\06_final_drift_original_frames.csv"

    df = pd.read_csv(drift_est_new_file)
    frames_continous = df['continuous_frame'].values * 1.61
    drift_x_nm = df['dx_nm'].values
    drift_y_nm = df['dy_nm'].values
    drift_z_nm = df['dz_nm'].values

    drift_est_old_file = folder_new + "comet_drift_estimate_full.csv"

    df = pd.read_csv(drift_est_old_file)
    frames_continous_old = df['frames'].values
    drift_x_nm_old = df['X [nm]'].values
    drift_y_nm_old = df['Y [nm]'].values
    drift_z_nm_old = df['Z [nm]'].values

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    ax[0].plot(frames_continous, drift_x_nm, color='blue', label='X Drift', linewidth=2)
    ax[0].plot(frames_continous_old, drift_x_nm_old, color='blue', linestyle='--', label='X Drift (Old)', linewidth=2)
    ax[0].set_ylabel('X Drift [nm]')

    ax[1].plot(frames_continous, drift_y_nm, color='orange', label='Y Drift', linewidth=2)
    ax[1].plot(frames_continous_old, drift_y_nm_old, color='orange', linestyle='--', label='Y Drift (Old)', linewidth=2)
    ax[1].set_ylabel('Y Drift [nm]')

    ax[2].plot(frames_continous, drift_z_nm, color='green', label='Z Drift', linewidth=2)
    ax[2].plot(frames_continous_old, drift_z_nm_old, color='green', linestyle='--', label='Z Drift (Old)', linewidth=2)
    ax[2].set_ylabel('Z Drift [nm]')

    plt.show()

if __name__ == "__main__":
    compare_new_pipeline_locs_basis()
    #compare_new_pipeline_drift_curve()
