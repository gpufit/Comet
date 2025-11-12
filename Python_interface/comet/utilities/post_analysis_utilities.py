from tkinter.filedialog import asksaveasfilename

import numpy as np
import h5py
from sympy import false


def combine_two_drift_estimates_from_correction_details_files(file_1, file_2, savename=None, sanity_check=False):
    if savename is None:
        savename = asksaveasfilename(defaultextension=".h5", title="Save combined drift as...")
    with h5py.File(file_1, 'r') as f1, h5py.File(file_2, 'r') as f2, h5py.File(savename, 'a') as fout:

        assert np.array_equal(f1['drift']['frames_interpolated'][:], f2['drift']['frames_interpolated'][:]), "Frame numbers do not match between the two files."
        fout.require_group('drift_correction_1')
        for key in f1:
            f1.copy(key, fout['drift_correction_1'])
        fout.require_group('drift_correction_2')
        for key in f2:
            f2.copy(key, fout['drift_correction_2'])

        drift_1 = f1['drift']['drift_nm'][:]
        drift_2 = f2['drift']['drift_nm'][:]

        combined_drift = (drift_1 + drift_2)
        fout.require_group('combined_drift')
        fout['combined_drift'].create_dataset('frames_interpolated', data=f1['drift']['frames_interpolated'][:])
        fout['combined_drift'].create_dataset('drift_nm', data=combined_drift)

        combined_drift_with_frames = np.zeros((combined_drift.shape[0], combined_drift.shape[1] + 1), dtype=combined_drift.dtype)
        combined_drift_with_frames[:, -1] = f1['drift']['frames_interpolated'][:]
        combined_drift_with_frames[:, :-1] = combined_drift

        if sanity_check:
            import matplotlib.pyplot as plt
            frames = combined_drift_with_frames[:, -1]
            plt.figure()
            plt.plot(frames, combined_drift_with_frames[:, 0], label='X Drift')
            plt.plot(frames, combined_drift_with_frames[:, 1], label='Y Drift')
            plt.plot(frames, combined_drift_with_frames[:, 2], label='Z Drift')
            plt.xlabel('Frame')
            plt.ylabel('Drift (nm)')
            plt.title('Combined Drift Estimate')
            plt.legend()
            plt.show()

    return combined_drift_with_frames