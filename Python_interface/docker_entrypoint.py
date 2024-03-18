import argparse
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import time
import warnings
import yaml
from cerberus import Validator
from typing import Literal, Optional
import drift_optimization_functions_2d, drift_optimization_functions_3d, segment_dataset


def load_param(param_path: str) -> dict:
    """Load parameters from file."""
    param = yaml.safe_load(open(param_path, "r"))
    schema = {
        "dataset_dimension": {"type": "integer", "min": 2, "max": 3},
        "segmentation_method": {"type": "string", "regex": "^(time|locs|frames)$"},
        "segmentation_parameter": {"type": "integer", "min": 1},
        "max_drift_nm": {"type": "float", "min": 0},
        "initial_gaussian_scale_nm": {"type": "float", "min": 0},
    }
    v = Validator(require_all=True)
    if not v.validate(param, schema):
        raise ValueError(v.errors)
    return param


def load_localizations(
    localizations_path: str,
) -> tuple[np.ndarray, Optional[tuple[int]]]:
    """Load localizations from file (csv or DECODE h5)."""
    file_ext = os.path.splitext(localizations_path)[1]
    if file_ext == ".csv":  # Comet format
        data = pd.read_csv(localizations_path).head(150000)
        localizations = np.zeros((len(data["frame"]), 4))
        for i, label in enumerate(["x [nm]", "y [nm]", "z [nm]", "frame"]):
            localizations[:, i] = np.asarray(data[label])
        px_size = None
    elif file_ext == ".h5":  # DECODE format
        with h5py.File(localizations_path, "r") as h5:
            data = {k: v for k, v in h5["data"].items() if v.shape is not None}
            data.update({k: None for k, v in h5["data"].items() if v.shape is None})
            meta_data = dict(h5["meta"].attrs)
            localizations = np.concatenate(
                (data["xyz"], np.expand_dims(data["frame_ix"], 1)), axis=1
            )
            if meta_data["xy_unit"] == "px":
                px_size = meta_data["px_size"]
                localizations[:, :2] = localizations[:, :2] * np.array(px_size)
    else:
        raise ValueError(
            f"Localizations file of type '{file_ext}' of file '{localizations_path}' not supported; please provide 'csv' or 'h5'."
        )
    return localizations, px_size


def get_segmentation(param: dict, localizations: np.ndarray) -> tuple[np.ndarray, int]:
    """Segment localizations."""
    segmentation_method = param["segmentation_method"]
    segmentation_parameter = param["segmentation_parameter"]
    if segmentation_method == "time":
        localizations, n_segments = segment_dataset.segment_by_num_windows(
            localizations, segmentation_parameter, True
        )
    elif segmentation_method == "locs":
        localizations, n_segments = segment_dataset.segment_by_num_locs_per_window(
            localizations, segmentation_parameter, True
        )
    elif segmentation_method == "frames":
        localizations, n_segments = segment_dataset.segment_by_frame_windows(
            localizations, segmentation_parameter, True
        )
    else:
        raise ValueError(f"segmentation method {segmentation_method} not recognized")
    return localizations, n_segments


def get_drift(param: dict, n_segments: int, localizations: np.ndarray) -> np.ndarray:
    """Estimate drift."""
    max_drift_nm = float(param["max_drift_nm"])
    dataset_dimension = param["dataset_dimension"]
    initial_gaussian_scale_nm = float(param["initial_gaussian_scale_nm"])
    warnings.filterwarnings("ignore")
    if dataset_dimension == 3:
        drift = drift_optimization_functions_3d.optimize_3d_chunked(
            n_segments,
            localizations,
            sigma_nm=initial_gaussian_scale_nm,
            drift_max_nm=max_drift_nm,
        )
        drift = drift.reshape(n_segments, 3)
    else:
        drift = drift_optimization_functions_2d.optimize_2d_chunked(
            n_segments,
            localizations,
            drift_max_nm=max_drift_nm,
            sigma_nm=initial_gaussian_scale_nm,
        )
        drift = drift.reshape(n_segments, 2)
    return drift


def plot_drift(
    drift: np.ndarray, n_segments: int, n_frames: int, dataset_dimension: Literal[2, 3]
) -> plt.Figure:
    """Plot drift estimate over frames."""
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_segments) / n_segments * n_frames, drift[:, 0])
    ax.plot(np.arange(n_segments) / n_segments * n_frames, drift[:, 1])
    if dataset_dimension == 3:
        ax.plot(np.arange(n_segments) / n_segments * n_frames, drift[:, 2])
    ax.set(xlabel="Segments", ylabel="Drift estimate [nm]")
    ax.legend(
        ["x est.", "y est.", "z est."]
        if dataset_dimension == 3
        else ["x est.", "y est."]
    )
    return fig


def save_drift(
    dataset_dimension: Literal[2, 3],
    n_segments: int,
    drift: np.ndarray,
    px_size: Optional[tuple[int]],
    output_path: str,
):
    """Save drift estimate to file (csv or h5)."""
    frames = np.arange(n_segments)  # frames
    file_ext = os.path.splitext(output_path)[1]
    if file_ext == ".csv":
        data = pd.DataFrame({"frame": frames, "x_nm": drift[:, 0], "y_nm": drift[:, 1]})
        if dataset_dimension == 3:
            data["z_nm"] = drift[:, 2]
        data.to_csv(output_path, index=False)
    elif file_ext == ".h5":
        with h5py.File(output_path, "w") as h5:
            data = h5.create_group("data")
            data.create_dataset("frame_ix", data=frames)
            data.create_dataset("xyz_nm", data=drift)
            meta = h5.create_group("meta")
            meta.attrs.update({"px_size": px_size})
    else:
        raise ValueError(f"Output file type must be '.csv' or '.h5', not '{file_ext}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--localizations_path")
    parser.add_argument("-p", "--param_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)  # prepare output directory
    param = load_param(args.param_path)
    print("Read parameters:", param)
    localizations, px_size = load_localizations(args.localizations_path)
    n_locs, n_frames = localizations.shape[0], len(np.unique(localizations[:, -1]))
    print(f"Loaded {n_locs} localizations from {n_frames} frames")

    localizations, n_segments = get_segmentation(param, localizations)
    print(f"Segmentation into {n_segments} segments done")
    time_start = time.time()
    drift = get_drift(param, n_segments, localizations)
    print(f"Drift estimation done in {time.time() - time_start:.2f} seconds")

    fig_drift = plot_drift(drift, n_segments, n_frames, param["dataset_dimension"])
    fig_drift.savefig(f"{args.output_path}/drift.png")
    output_path = (
        f"{args.output_path}/drift{os.path.splitext(args.localizations_path)[1]}"
    )
    save_drift(param["dataset_dimension"], n_segments, drift, px_size, output_path)
