from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_INTERFACE = PROJECT_ROOT / "Python_interface"
if str(PYTHON_INTERFACE) not in sys.path:
    sys.path.insert(0, str(PYTHON_INTERFACE))

try:
    from comet.core.io_utils import save_dataset_as_ms_h5
except ModuleNotFoundError:
    from Python_interface.comet.core.io_utils import save_dataset_as_ms_h5

try:
    from oligostorm_comet_pipeline import _save_extra_data
except ModuleNotFoundError:
    from process_oligostorm_data.oligostorm_comet_pipeline import _save_extra_data


INPUT_CSV = ""
DRIFT_CSV = ""
OUTPUT_MOLECULE_SET = ""
OUTPUT_DRIFT_CORRECTED_CSV = ""

X_COLUMN = "x"
Y_COLUMN = "y"
Z_COLUMN = "z"
FRAME_COLUMN = "frame"
TIMEPOINT_COLUMN = "time-point"

PHOTON_COLUMN = "photon-count"
UNCERTAINTY_X_COLUMN = "precisionx"
UNCERTAINTY_Y_COLUMN = "precisiony"
UNCERTAINTY_Z_COLUMN = "precisionz"

PIXEL_SIZE_NM = 160
PIXEL_SIZE_Z_NM = 160
DRIFT_TIMEPOINT_FILTER = None
INTERPOLATE_MISSING_FRAMES = True
KEEP_UNMATCHED_TIMEPOINTS = True


def _require_columns(df, columns, file_label):
    missing = [column for column in columns if column is not None and column not in df.columns]
    if missing:
        raise KeyError(f"{file_label} is missing required columns: {missing}")


def _optional_array(df, column):
    if column is None or column not in df.columns:
        return None
    return df[column].to_numpy(dtype=float)


def load_original_csv_as_dataset(
    input_csv,
    x_column=X_COLUMN,
    y_column=Y_COLUMN,
    z_column=Z_COLUMN,
    frame_column=FRAME_COLUMN,
    timepoint_column=TIMEPOINT_COLUMN,
):
    df = pd.read_csv(input_csv)
    _require_columns(df, [x_column, y_column, z_column, frame_column], "input CSV")

    dataset = np.column_stack(
        [
            df[x_column].to_numpy(dtype=float),
            df[y_column].to_numpy(dtype=float),
            df[z_column].to_numpy(dtype=float),
            np.rint(df[frame_column].to_numpy(dtype=float)).astype(np.int64),
        ]
    )
    if timepoint_column is not None and timepoint_column in df.columns:
        dataset = np.column_stack(
            [dataset, np.rint(df[timepoint_column].to_numpy(dtype=float)).astype(np.int64)]
        )
    return dataset, df


def save_original_csv_as_molecule_set(
    input_csv,
    output_h5=None,
    x_column=X_COLUMN,
    y_column=Y_COLUMN,
    z_column=Z_COLUMN,
    frame_column=FRAME_COLUMN,
    timepoint_column=TIMEPOINT_COLUMN,
    photon_column=PHOTON_COLUMN,
    uncertainty_x_column=UNCERTAINTY_X_COLUMN,
    uncertainty_y_column=UNCERTAINTY_Y_COLUMN,
    uncertainty_z_column=UNCERTAINTY_Z_COLUMN,
    pixelsize_nm=PIXEL_SIZE_NM,
    pixelsize_z_nm=PIXEL_SIZE_Z_NM,
):
    input_csv = Path(input_csv)
    if output_h5 is None or output_h5 == "":
        output_h5 = input_csv.with_suffix(".molecule_set.h5")
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    dataset, df = load_original_csv_as_dataset(
        input_csv,
        x_column=x_column,
        y_column=y_column,
        z_column=z_column,
        frame_column=frame_column,
        timepoint_column=timepoint_column,
    )

    # save_dataset_as_ms_h5 writes X from column 1 and Y from column 0.
    # Passing y,x,z here makes molecule-set reloads preserve the original CSV x,y,z columns.
    writer_coordinates = dataset[:, [1, 0, 2]]
    save_dataset_as_ms_h5(
        writer_coordinates,
        dataset[:, 3].astype(np.int64),
        pixelsize_nm=pixelsize_nm,
        pixelsize_z_nm=pixelsize_z_nm,
        amplitudes=_optional_array(df, photon_column),
        uncertainty_x=_optional_array(df, uncertainty_x_column),
        uncertainty_y=_optional_array(df, uncertainty_y_column),
        uncertainty_z=_optional_array(df, uncertainty_z_column),
        filename=str(output_h5),
    )
    _save_extra_data(output_h5, {"source_csv": str(input_csv)})

    if dataset.shape[1] > 4:
        with h5py.File(output_h5, "a") as handle:
            handle["molecule_set_data"]["datatable"]["CHANNEL"] = dataset[:, 4].astype(np.int32)

    return output_h5


def _collapse_duplicate_frame_rows(drift_df):
    return (
        drift_df[["original_frame", "dx_nm", "dy_nm", "dz_nm"]]
        .groupby("original_frame", as_index=False)
        .mean()
        .sort_values("original_frame")
    )


def _drift_for_frames(drift_df, frames, interpolate_missing=True):
    drift_df = _collapse_duplicate_frame_rows(drift_df)
    source_frames = drift_df["original_frame"].to_numpy(dtype=float)
    drift_values = drift_df[["dx_nm", "dy_nm", "dz_nm"]].to_numpy(dtype=float)
    target_frames = np.asarray(frames, dtype=float)

    if len(source_frames) == 0:
        return None
    if len(source_frames) == 1:
        return np.repeat(drift_values, len(target_frames), axis=0)

    if not interpolate_missing:
        indexed = drift_df.set_index("original_frame")
        missing = sorted(set(target_frames.astype(int)) - set(indexed.index.astype(int)))
        if missing:
            raise ValueError(f"CSV contains frames absent from drift table, e.g. {missing[:10]}")
        return indexed.loc[target_frames.astype(int), ["dx_nm", "dy_nm", "dz_nm"]].to_numpy(dtype=float)

    return np.column_stack(
        [
            np.interp(target_frames, source_frames, drift_values[:, axis])
            for axis in range(3)
        ]
    )


def apply_final_drift_to_original_csv(
    input_csv,
    drift_csv,
    output_csv=None,
    x_column=X_COLUMN,
    y_column=Y_COLUMN,
    z_column=Z_COLUMN,
    frame_column=FRAME_COLUMN,
    timepoint_column=TIMEPOINT_COLUMN,
    drift_timepoint_filter=DRIFT_TIMEPOINT_FILTER,
    interpolate_missing_frames=INTERPOLATE_MISSING_FRAMES,
    keep_unmatched_timepoints=KEEP_UNMATCHED_TIMEPOINTS,
):
    input_csv = Path(input_csv)
    drift_csv = Path(drift_csv)
    if output_csv is None or output_csv == "":
        output_csv = input_csv.with_name(f"{input_csv.stem}_drift_corrected.csv")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    drift_df = pd.read_csv(drift_csv)
    _require_columns(df, [x_column, y_column, z_column, frame_column], "input CSV")
    _require_columns(drift_df, ["original_frame", "dx_nm", "dy_nm", "dz_nm", "timepoint_label"], "drift CSV")

    corrected = df.copy()
    frames = corrected[frame_column].to_numpy(dtype=int)
    unmatched_timepoints = []

    if timepoint_column is not None and timepoint_column in corrected.columns:
        labels = np.rint(corrected[timepoint_column].to_numpy(dtype=float)).astype(np.int64)
        for label in np.unique(labels):
            mask = labels == label
            drift_group = drift_df[drift_df["timepoint_label"].astype(np.int64) == int(label)]
            drift_values = _drift_for_frames(drift_group, frames[mask], interpolate_missing_frames)
            if drift_values is None:
                unmatched_timepoints.append(int(label))
                if keep_unmatched_timepoints:
                    continue
                raise ValueError(f"No drift rows found for timepoint label {label}.")
            corrected.loc[mask, x_column] = corrected.loc[mask, x_column].to_numpy(dtype=float) - drift_values[:, 0]
            corrected.loc[mask, y_column] = corrected.loc[mask, y_column].to_numpy(dtype=float) - drift_values[:, 1]
            corrected.loc[mask, z_column] = corrected.loc[mask, z_column].to_numpy(dtype=float) - drift_values[:, 2]
    else:
        if drift_timepoint_filter is not None:
            drift_df = drift_df[drift_df["timepoint_label"].astype(np.int64) == int(drift_timepoint_filter)]
        elif drift_df.duplicated("original_frame").any():
            raise ValueError(
                "Drift CSV has multiple timepoints per original frame. Set TIMEPOINT_COLUMN "
                "or DRIFT_TIMEPOINT_FILTER before applying it to this CSV."
            )
        drift_values = _drift_for_frames(drift_df, frames, interpolate_missing_frames)
        if drift_values is None:
            raise ValueError("No drift rows available after filtering.")
        corrected[x_column] = corrected[x_column].to_numpy(dtype=float) - drift_values[:, 0]
        corrected[y_column] = corrected[y_column].to_numpy(dtype=float) - drift_values[:, 1]
        corrected[z_column] = corrected[z_column].to_numpy(dtype=float) - drift_values[:, 2]

    corrected.to_csv(output_csv, index=False, float_format="%.6f")
    if unmatched_timepoints:
        print(f"Kept unmatched timepoint label(s) uncorrected: {unmatched_timepoints}")
    return output_csv


if __name__ == "__main__":
    if INPUT_CSV and OUTPUT_MOLECULE_SET is not None:
        saved_h5 = save_original_csv_as_molecule_set(INPUT_CSV, OUTPUT_MOLECULE_SET)
        print(f"Saved molecule set to: {saved_h5}")

    if INPUT_CSV and DRIFT_CSV:
        saved_csv = apply_final_drift_to_original_csv(
            input_csv=INPUT_CSV,
            drift_csv=DRIFT_CSV,
            output_csv=OUTPUT_DRIFT_CORRECTED_CSV,
        )
        print(f"Saved drift-corrected CSV to: {saved_csv}")
