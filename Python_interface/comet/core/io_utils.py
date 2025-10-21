import numpy as np
import pandas as pd
import h5py
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import matplotlib.pyplot as plt
import csv


def load_thunderstorm_csv(filename=None, return_essentials=True):
    """
    Load a ThunderSTORM CSV file and return the localization data.
    Parameters:
    - filename: str, path to the CSV file. If None, a file dialog will open.
    - return_essentials: bool, if True, return only essential columns (x, y, z, frame).
    Returns:
    - np.ndarray or pd.DataFrame: localization data.
    """
    if filename is None:
        Tk().withdraw()
        filename = askopenfilename(title="Select ThunderSTORM CSV file")

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return -1

    if not return_essentials:
        return df

    keys = df.columns.to_numpy()
    print(f"Opened ThunderSTORM file with keys: {keys}")

    # Initialise locs information with following convention:
    # Column 0: x coordinates
    # Column 1: y coordinates
    # Column 0: z coordinates
    # Column 0: time frames
    locs = np.zeros((len(df), 4))

    locs[:, 0] = df["x [nm]"]
    locs[:, 1] = df["y [nm]"]
    # Fill the z coordinates in 3rd column if available, leave full of 0's otherwise
    if "z [nm]" in keys:
        locs[:, 2] = df["z [nm]"]
    locs[:, 3] = df["frame"]

    return locs


def load_normal_molecule_set(filename=None, sanity_check=False, photon_bandpass=(None, None)):
    """
    Load a normal molecule set from an HDF5 file.
    Parameters:
    - filename: str, path to the HDF5 file. If None, a file dialog will open.
    - sanity_check: bool, if True, display a scatter plot of the XY positions.
    - photon_bandpass: tuple, (min_photons, max_photons) to filter localizations by photon count.
    Returns:
    - np.ndarray: localization data with columns [x_nm, y_nm, z_nm, frame].
    """
    if filename is None:
        Tk().withdraw()
        filename = askopenfilename(title="Select HDF5 dataset")

    f = h5py.File(filename, 'r')
    try:
        g = f['molecule_set_data']
        if 'pixel_size_um' in g:
            pixelsize_nm = np.asarray(g['pixel_size_um']) * 1e3
            pixelsize_z_nm = pixelsize_nm
        elif 'xy_pixel_size_um' in g and 'z_pixel_size_um' in g:
            pixelsize_nm = np.asarray(g['xy_pixel_size_um']) * 1e3
            pixelsize_z_nm = np.asarray(g['z_pixel_size_um']) * 1e3
        elif 'pixelsize_um' in g:
            pixelsize_nm = np.asarray(g['pixelsize_um']) * 1e3
            pixelsize_z_nm = pixelsize_nm
        else:
            raise KeyError("No valid pixel size keys found.")
    except Exception as e:
        raise RuntimeError(f"Failed to read pixel size: {e}")

    datatable = f['molecule_set_data']['datatable']
    x_pos = np.asarray(datatable['X_POS_PIXELS']) * pixelsize_nm
    y_pos = np.asarray(datatable['Y_POS_PIXELS']) * pixelsize_nm
    z_pos = np.asarray(datatable['Z_POS_PIXELS']) * pixelsize_z_nm
    frames = np.asarray(datatable['FRAME_NUMBER'])

    locs = np.stack([x_pos, y_pos, z_pos, frames], axis=1)

    if photon_bandpass[0] is not None and photon_bandpass[1] is not None:
        photons = np.asarray(datatable['PHOTONS'])
        mask = (photons > photon_bandpass[0]) & (photons < photon_bandpass[1])
        locs = locs[mask]

    if sanity_check:
        plt.figure()
        plt.scatter(locs[:, 0], locs[:, 1], alpha=0.01)
        plt.title("Sanity Check: XY Projection")
        plt.show()

    return locs


def load_simulation_dataset_and_gt_drift(filename=None, display=False, remove_loc_prec=False):
    if filename is None:
        filename = askopenfilename(initialdir="..\\data\\")
    f = h5py.File(filename)
    frames = np.asarray(f['frame_number'], dtype=int)
    drift = np.asarray(f['sample_drift']['drift_data']) * 1E3
    if not remove_loc_prec:
        x = np.asarray(f['x_coords']) * 1E3
        y = np.asarray(f['y_coords']) * 1E3
        z = np.asarray(f['z_coords']) * 1E3
    else:
        label_sites_nm = np.asarray(f['label_sites']['site_data']) * 1E3
        label_site_idc = np.asarray(f['label_site_index'])
        x = label_sites_nm[0, label_site_idc]
        y = label_sites_nm[1, label_site_idc]
        z = label_sites_nm[2, label_site_idc]
        x += drift[0, frames]
        y += drift[1, frames]
        z += drift[2, frames]

    f.close()
    print(f"Imported {len(x)} coords on {frames.max() + 1} frames with a max drift of {np.max(np.abs(drift))} nm")
    if display:
        plt.figure()
        plt.scatter(x[::10], y[::10], alpha=0.01)
        plt.figure()
        plt.plot(drift[0, :])
        plt.plot(drift[1, :])
        plt.plot(drift[2, :])
        plt.show()
    dataset = np.zeros((len(x), 4))
    dataset[:, 0] = x
    dataset[:, 1] = y
    dataset[:, 2] = z
    dataset[:, 3] = frames
    drift = np.swapaxes(drift, 0, 1)
    return dataset, drift


def save_drift_correction_details(savename, drift_est, drift_est_intp, frames_intp,
                                  segmentation_result, calc_time, initial_sigma_nm, target_sigma_nm, gt_drift=None):
    """
    Save drift correction details to an HDF5 file.
    Parameters:
    - savename: str, path to save the HDF5 file.
    - drift_est: np.ndarray, estimated drift per segment in nm.
    - drift_est_intp: np.ndarray, interpolated drift estimates in nm.
    - frames_intp: np.ndarray, frames corresponding to the interpolated drift estimates.
    - segmentation_result: SegmentationResult, result of the segmentation process.
    - calc_time: float, time taken for the drift calculation in seconds.
    - initial_sigma_nm: float, initial sigma used in drift estimation.
    - target_sigma_nm: float, target sigma used in drift estimation.
    - gt_drift: np.ndarray or None, ground truth drift if available.
    """
    f = h5py.File(savename, 'a')

    f.create_group("drift")
    f['drift']['center_frames'] = segmentation_result.center_frames
    f['drift']['drift_per_segment_nm'] = drift_est
    f['drift']['drift_interpolated_nm'] = drift_est_intp
    f['drift']['frames_interpolated'] = frames_intp
    if gt_drift is not None:
        f['drift']['gt_drift'] = gt_drift
    f.create_group('parameters')
    f['parameters']['calculation_time_s'] = calc_time
    for key, value in segmentation_result.out_dict.items():
        f['parameters'][key] = value
    f['parameters']['initial_sigma_nm'] = initial_sigma_nm
    f['parameters']['target_sigma_nm'] = target_sigma_nm
    f.close()


def save_dataset_as_ms_h5(storm_coordinates, frames, pixelsize_nm, pixelsize_z_nm=None, amplitudes=None,
                                 uncertainty_x=None, uncertainty_y=None, uncertainty_z=None, frame_shape=None,
                                 filename=None, extra_dict=None):
    """
    Save dataset in Molecule Set format used by Daxview.
    Parameters:
    - storm_coordinates: np.ndarray, shape (N, 2) or (N, 3), coordinates in nm.
    - frames: np.ndarray, shape (N,), frame numbers.
    - pixelsize_nm: float, pixel size in nm.
    - pixelsize_z_nm: float or None, pixel size in z in nm. If None, set equal to pixelsize_nm.
    - amplitudes: np.ndarray or None, shape (N,), photon counts. If None, set to 1000.
    - uncertainty_x: np.ndarray or None, shape (N,), uncertainty in x in nm. If None, set to 10 nm.
    - uncertainty_y: np.ndarray or None, shape (N,), uncertainty in y in nm. If None, set to 10 nm.
    - uncertainty_z: np.ndarray or None, shape (N,), uncertainty in z in nm. If None, set to 10 nm.
    - frame_shape: tuple or None, shape of the frames (height, width). If None, set to (128, 128).
    - filename: str or None, path to save the HDF5 file. If None, a file dialog will open.
    - extra_dict: dict or None, additional key-value pairs to save in the file.
    """
    if filename is None:
        filename = asksaveasfilename(defaultextension=".h5", filetypes=[("hdf5 files", "*.h5")])
    f = h5py.File(filename, 'w')

    headers = ["X_POS_PIXELS", "Y_POS_PIXELS", "Z_POS_PIXELS", "PRECISION_XY_PIXELS", "PRECISION_Z_PIXELS",
               "PHOTONS", "CHANNEL", "FRAME_NUMBER", "INDEX"]
    dtypes = [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.int32, np.int32, np.int32]
    compound_dtype = np.dtype([(headers[i], dtypes[i]) for i in range(len(headers))])

    structured_array = np.zeros(len(frames), dtype=compound_dtype)
    structured_array['X_POS_PIXELS'] = storm_coordinates[:, 1] / pixelsize_nm
    structured_array['Y_POS_PIXELS'] = storm_coordinates[:, 0] / pixelsize_nm
    if storm_coordinates.shape[1] > 2:
        structured_array['Z_POS_PIXELS'] = storm_coordinates[:, 2] / pixelsize_nm
    else:
        structured_array['Z_POS_PIXELS'] = np.ones(len(frames))
    if amplitudes is not None:
        structured_array['PHOTONS'] = amplitudes
    else:
        structured_array['PHOTONS'] = np.ones(len(frames)) * 1E3
    if uncertainty_x is not None:
        structured_array['PRECISION_XY_PIXELS'] = np.sqrt(uncertainty_x ** 2 + uncertainty_y ** 2) / pixelsize_nm
    else:
        structured_array['PRECISION_XY_PIXELS'] = np.ones(len(frames)) / 10
    if uncertainty_z is not None:
        structured_array['PRECISION_Z_PIXELS'] = uncertainty_z / pixelsize_nm
    else:
        structured_array['PRECISION_Z_PIXELS'] = np.ones(len(frames)) / 10
    structured_array['CHANNEL'] = np.zeros(len(frames))
    structured_array['FRAME_NUMBER'] = frames
    structured_array['INDEX'] = np.arange(len(frames))

    f['daxview_file_type'] = np.array('DAXVIEW_STORM_DATA', dtype=h5py.string_dtype('ascii', 19))
    f['object_class_name'] = np.array('DV_STORM_4PI_MOLECULE_SET', dtype=h5py.string_dtype('ascii', 26))
    f.create_dataset('save_format_version', data=np.ones(1, dtype=np.float32) * 1.2)
    f.create_group('molecule_set_data')
    f['molecule_set_data'].create_dataset('datatable', data=structured_array)
    f['molecule_set_data']['flag_dcorr_applied'] = 1
    f['molecule_set_data']['flag_tform_applied'] = 0
    f['molecule_set_data']['n_keys'] = 6
    f['molecule_set_data']['keynames'] = np.asarray(["z_pixel_size_um", "xy_pixel_size_um", "version", "n_molecules",
                                                     "flag_dcorr_applied", "xpixels", "stat_t_matrix", "ypixels",
                                                     "flag_tform_applied", "datatable"],
                                                    dtype=h5py.string_dtype('ascii', 19))
    f['molecule_set_data']['n_molecules'] = len(frames)
    f['molecule_set_data']['stat_t_matrix'] = np.diag([1, 1, 1, 1])
    f['molecule_set_data']['version'] = 1
    if frame_shape is None:
        f['molecule_set_data']['xpixels'] = 128
        f['molecule_set_data']['ypixels'] = 128
    else:
        f['molecule_set_data']['xpixels'] = frame_shape[0]
        f['molecule_set_data']['ypixels'] = frame_shape[1]
    f['molecule_set_data']['xy_pixel_size_um'] = pixelsize_nm / 1E3
    if pixelsize_z_nm is None:
        f['molecule_set_data']['z_pixel_size_um'] = pixelsize_nm / 1E3
    else:
        f['molecule_set_data']['z_pixel_size_um'] = pixelsize_z_nm / 1E3

    if extra_dict is not None:
        group = f.create_group('extra_data')
        for key, value in extra_dict.items():
            group[key] = value
    f.close()


def save_dataset_as_thunderstorm_csv(dataset, savename=None):
    """
    Save dataset in ThunderSTORM CSV format.
    Parameters:
    - dataset: np.ndarray, shape (N, 4), localization data with columns [x_nm, y_nm, z_nm, frame].
    - savename: str or None, path to save the CSV file. If None, a file dialog will open.
    """
    if savename is None:
        Tk().withdraw()
        savename = asksaveasfilename(title="Save as ThunderSTORM CSV", defaultextension=".csv")

    df = pd.DataFrame({
        "frame": dataset[:, 3].astype(int),
        "x [nm]": dataset[:, 0],
        "y [nm]": dataset[:, 1],
    })

    header = ['"frame"', '"x [nm]"', '"y [nm]"']
    # Add z coordinates if present
    if (np.any(dataset[:, 2] != np.zeros(len(dataset)))):
        df["z [nm]"] = dataset[:, 2]
        header.append('"z [nm]"')

    df.to_csv(savename, index=True,
              header=header, index_label='"id"',
              mode='w', quoting=csv.QUOTE_NONE, 
              float_format = '%.3f') # precision of 3 


def save_dataset_custom(dataset, savename=None, format_hint=None):
    """
    Placeholder function. Extend this to support custom save formats.
    Use 'format_hint' to route specific formats.
    Parameters:
    - dataset: np.ndarray, localization data.
    - savename: str or None, path to save the file. If None, a file dialog will open.
    - format_hint: str or None, hint for the desired format.
    """
    raise NotImplementedError("Custom save format handler not implemented. Use HDF5 or ThunderSTORM CSV.")
