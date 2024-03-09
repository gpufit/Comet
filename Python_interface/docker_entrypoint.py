import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import time
import warnings
import yaml
from cerberus import Validator
import drift_optimization_functions_2d, drift_optimization_functions_3d, segment_dataset


def run(param: dict, localizations: np.ndarray, output_path: str) -> np.ndarray:
    # segment
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
    print(f"segmentation into {n_segments} segments done")

    # estimate drift
    max_drift_nm = float(param["max_drift_nm"])
    dataset_dimension = param["dataset_dimension"]
    initial_gaussian_scale_nm = float(param["initial_gaussian_scale_nm"])
    t = time.time()
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
    print(f"drift estimation done in {np.round(time.time()-t)}s")

    # plot drift
    frames = np.unique(localizations[:, -1])
    n_frames = len(frames)
    fig_drift = plot_drift(drift, n_segments, n_frames, dataset_dimension)
    fig_drift.savefig(f"{output_path}/drift_estimate.png")

    # save drift
    frames = np.arange(n_segments)
    result = np.zeros((len(drift[:, 0]), dataset_dimension + 1))
    result[:, 0] = frames
    for i in range(dataset_dimension):
        result[:, i + 1] = drift[:, i]
    header = "frame,x_nm,y_nm" + (",z_nm" if dataset_dimension == 3 else "") + "\n"
    file_contents = (
        header
        + str(result.tolist())
        .replace(" [", "")
        .replace("[", "")
        .replace("],", "\n")
        .replace("]", "")
    ).encode()
    with open(f"{output_path}/drift_estimate.csv", "wb+") as f:
        f.write(file_contents)


def plot_drift(drift, n_segments, n_frames, dataset_dimension):
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


def _load_param(param_path: str) -> dict:
    param = yaml.safe_load(open(param_path, "r"))
    # validate
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
    print("validated parameters:", param)
    return param


def _load_localizations(localizations_path: str) -> np.ndarray:
    data = pd.read_csv(localizations_path).head(150000)
    localizations = np.zeros((len(data["frame"]), 4))
    localizations[:, 0] = np.asarray(data["x [nm]"])
    localizations[:, 1] = np.asarray(data["y [nm]"])
    localizations[:, 2] = np.asarray(data["z [nm]"])
    localizations[:, 3] = np.asarray(data["frame"])
    frames = np.unique(localizations[:, -1])
    n_frames = len(frames)
    print(
        f"{localizations_path} import successful, {len(localizations[:, 0])} localizations, {n_frames} frames"
    )
    return localizations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--localizations_path")
    parser.add_argument("-p", "--param_path")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()

    param = _load_param(args.param_path)
    localizations = _load_localizations(args.localizations_path)
    print("doing something")

    os.makedirs(args.output_path, exist_ok=True)
    run(param, localizations, args.output_path)
