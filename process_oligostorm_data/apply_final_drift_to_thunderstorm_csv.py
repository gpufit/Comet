from pathlib import Path

import numpy as np
import pandas as pd


FRAME_COLUMN = "frame"
X_COLUMN = "x [nm]"
Y_COLUMN = "y [nm]"
Z_COLUMN = "z [nm]"  # Set to None for 2D CSVs.

# Set this if the CSV has a column matching the pipeline timepoint/channel labels.
TIMEPOINT_COLUMN = None

# Use this when applying one timepoint from a multi-timepoint drift CSV to a CSV
# without a timepoint column, e.g. DRIFT_TIMEPOINT_FILTER = 0.
DRIFT_TIMEPOINT_FILTER = None

INTERPOLATE_MISSING_FRAMES = True


def _require_columns(df, columns, file_label):
    missing = [column for column in columns if column is not None and column not in df.columns]
    if missing:
        raise KeyError(f"{file_label} is missing required columns: {missing}")


def _collapse_duplicate_frame_rows(drift_df):
    drift_columns = ["dx_nm", "dy_nm", "dz_nm"]
    return (
        drift_df[["original_frame", *drift_columns]]
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
        raise ValueError("No drift rows available after filtering.")
    if len(source_frames) == 1:
        return np.repeat(drift_values, len(target_frames), axis=0)

    if not interpolate_missing:
        indexed = drift_df.set_index("original_frame")
        missing = sorted(set(target_frames.astype(int)) - set(indexed.index.astype(int)))
        if missing:
            raise ValueError(f"CSV contains frames absent from drift table, e.g. {missing[:10]}")
        return indexed.loc[target_frames.astype(int), ["dx_nm", "dy_nm", "dz_nm"]].to_numpy(dtype=float)

    interpolated = np.column_stack(
        [
            np.interp(target_frames, source_frames, drift_values[:, axis])
            for axis in range(3)
        ]
    )
    return interpolated


def apply_final_drift_to_csv(
    input_csv,
    drift_csv,
    output_csv=None,
    frame_column=FRAME_COLUMN,
    x_column=X_COLUMN,
    y_column=Y_COLUMN,
    z_column=Z_COLUMN,
    timepoint_column=TIMEPOINT_COLUMN,
    drift_timepoint_filter=DRIFT_TIMEPOINT_FILTER,
    interpolate_missing_frames=INTERPOLATE_MISSING_FRAMES,
):
    input_csv = Path(input_csv)
    drift_csv = Path(drift_csv)
    if output_csv is None or output_csv == "":
        output_csv = input_csv.with_name(f"{input_csv.stem}_drift_corrected.csv")
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)
    drift_df = pd.read_csv(drift_csv)

    _require_columns(df, [frame_column, x_column, y_column, z_column, timepoint_column], "input CSV")
    _require_columns(drift_df, ["original_frame", "dx_nm", "dy_nm", "dz_nm", "timepoint_label"], "drift CSV")

    corrected = df.copy()
    corrected["_drift_dx_nm"] = 0.0
    corrected["_drift_dy_nm"] = 0.0
    corrected["_drift_dz_nm"] = 0.0

    if timepoint_column is not None:
        for timepoint_label, row_indices in corrected.groupby(timepoint_column).groups.items():
            drift_group = drift_df[drift_df["timepoint_label"].astype(int) == int(timepoint_label)]
            if drift_group.empty:
                raise ValueError(f"No drift rows found for timepoint label {timepoint_label}.")
            frames = corrected.loc[row_indices, frame_column].to_numpy(dtype=int)
            drift_values = _drift_for_frames(drift_group, frames, interpolate_missing_frames)
            corrected.loc[row_indices, ["_drift_dx_nm", "_drift_dy_nm", "_drift_dz_nm"]] = drift_values
    else:
        if drift_timepoint_filter is not None:
            drift_df = drift_df[drift_df["timepoint_label"].astype(int) == int(drift_timepoint_filter)]
            if drift_df.empty:
                raise ValueError(f"No drift rows found for timepoint label {drift_timepoint_filter}.")
        elif drift_df.duplicated("original_frame").any():
            raise ValueError(
                "Drift CSV has multiple timepoints per original frame. Set TIMEPOINT_COLUMN "
                "or DRIFT_TIMEPOINT_FILTER before applying it to this CSV."
            )

        drift_values = _drift_for_frames(
            drift_df,
            corrected[frame_column].to_numpy(dtype=int),
            interpolate_missing_frames,
        )
        corrected[["_drift_dx_nm", "_drift_dy_nm", "_drift_dz_nm"]] = drift_values

    corrected[x_column] = corrected[x_column] - corrected["_drift_dx_nm"]
    corrected[y_column] = corrected[y_column] - corrected["_drift_dy_nm"]
    if z_column is not None and z_column in corrected.columns:
        corrected[z_column] = corrected[z_column] - corrected["_drift_dz_nm"]

    corrected = corrected.drop(columns=["_drift_dx_nm", "_drift_dy_nm", "_drift_dz_nm"])
    corrected.to_csv(output_csv, index=False, float_format="%.6f")
    return output_csv


if __name__ == "__main__":
    folder = r""
    input_filename = ""
    drift_filename = ""
    output_filename = ""

    saved_path = apply_final_drift_to_csv(
        input_csv=input_filename,
        drift_csv=drift_filename,
        output_csv=output_filename,
    )
    print(f"Saved drift-corrected CSV to: {saved_path}")
