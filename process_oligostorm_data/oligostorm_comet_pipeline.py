from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, RectangleSelector
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label, maximum_filter
import matplotlib as mpl
mpl.use("TkAgg")  # Use Tk backend for interactive features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_INTERFACE = PROJECT_ROOT / "Python_interface"
if str(PYTHON_INTERFACE) not in sys.path:
    sys.path.insert(0, str(PYTHON_INTERFACE))

try:
    from comet.core.io_utils import (
        load_normal_molecule_set,
        load_thunderstorm_csv,
        save_dataset_as_ms_h5,
    )
except ModuleNotFoundError:
    from Python_interface.comet.core.io_utils import (
        load_normal_molecule_set,
        load_thunderstorm_csv,
        save_dataset_as_ms_h5,
    )

_COMET_IMPORT_ERROR = None
try:
    from comet.core.drift_optimizer import comet_run_kd as _comet_run_kd
except ModuleNotFoundError as exc:
    _comet_run_kd = None
    _COMET_IMPORT_ERROR = exc

try:
    from Python_interface.CrossCorrelation.rendering.gpu_gaussian_rendering import (
        gpu_gaussian_render_2d as _gpu_gaussian_render_2d,
    )
except ModuleNotFoundError:
    _gpu_gaussian_render_2d = None


def _get_comet_run_kd():
    global _comet_run_kd, _COMET_IMPORT_ERROR
    if _comet_run_kd is not None:
        return _comet_run_kd
    try:
        from comet.core.drift_optimizer import comet_run_kd as imported
    except ModuleNotFoundError as exc:
        _COMET_IMPORT_ERROR = exc
        raise ModuleNotFoundError(
            "COMET drift correction dependencies are not importable in this Python environment. "
            "Install the Python_interface package dependencies or run inside the project venv."
        ) from exc
    _comet_run_kd = imported
    _COMET_IMPORT_ERROR = None
    return _comet_run_kd


def _ask_open(title="Select dataset"):
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw()
    return askopenfilename(
        title=title,
        filetypes=[("Localization datasets", "*.h5 *.hdf5 *.csv"), ("All files", "*.*")],
    )


def _ask_save(title="Save dataset", defaultextension=".h5"):
    from tkinter import Tk
    from tkinter.filedialog import asksaveasfilename

    Tk().withdraw()
    return asksaveasfilename(title=title, defaultextension=defaultextension)


def _load_molecule_set_with_channel(filename, sanity_check=False):
    dataset = load_normal_molecule_set(str(filename), sanity_check=sanity_check)
    with h5py.File(filename, "r") as handle:
        datatable = handle["molecule_set_data"]["datatable"]
        if "CHANNEL" not in datatable.dtype.names:
            return dataset
        channel = np.asarray(datatable["CHANNEL"], dtype=np.float64)
    return np.column_stack((dataset, channel))


def _load_dataset(dataset_or_path=None):
    if dataset_or_path is None:
        dataset_or_path = _ask_open()

    if isinstance(dataset_or_path, (str, Path)):
        path = Path(dataset_or_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return load_thunderstorm_csv(str(path)), path
        if suffix in {".h5", ".hdf5"}:
            return _load_molecule_set_with_channel(path), path
        raise ValueError(f"Unsupported dataset suffix: {suffix}")

    return dataset_or_path, None


def _dataset_from_result(value):
    if isinstance(value, dict) and "dataset" in value:
        return value["dataset"]
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], np.ndarray):
        return value[0]
    return value


def _validate_dataset(dataset, name="dataset", copy=True):
    dataset, _ = _load_dataset(_dataset_from_result(dataset))
    arr = np.asarray(dataset, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"{name} must have shape (N, >=4) with columns [x_nm, y_nm, z_nm, frame, ...].")
    arr = arr.copy() if copy else arr
    finite = np.isfinite(arr).all(axis=1)
    if not np.any(finite):
        raise ValueError(f"{name} contains no finite localizations.")
    if not np.all(finite):
        arr = arr[finite].copy()
    rounded_frames = np.rint(arr[:, 3])
    if copy or not np.array_equal(arr[:, 3], rounded_frames):
        arr = arr.copy()
        arr[:, 3] = rounded_frames.astype(np.int64)
    return arr


def _write_h5_value(group, key, value):
    if value is None:
        return
    if key in group:
        del group[key]
    if isinstance(value, dict):
        subgroup = group.create_group(key)
        for subkey, subvalue in value.items():
            _write_h5_value(subgroup, str(subkey), subvalue)
        return
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        group.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding="utf-8"))
        return

    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        group.create_dataset(key, data=arr.astype(h5py.string_dtype(encoding="utf-8")))
    elif arr.dtype == object:
        group.create_dataset(key, data=str(value), dtype=h5py.string_dtype(encoding="utf-8"))
    else:
        group.create_dataset(key, data=arr)


def _save_extra_data(filename, extra_dict):
    if not extra_dict:
        return
    with h5py.File(filename, "a") as handle:
        group = handle.require_group("extra_data")
        for key, value in extra_dict.items():
            _write_h5_value(group, str(key), value)


def _save_dataset_h5(dataset, filename, pixelsize_nm=160, pixelsize_z_nm=None, extra_dict=None):
    if filename is None:
        return None
    dataset = _validate_dataset(dataset)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_as_ms_h5(
        dataset[:, :3],
        dataset[:, 3].astype(np.int64),
        pixelsize_nm=pixelsize_nm,
        pixelsize_z_nm=pixelsize_z_nm,
        filename=str(filename),
    )
    if dataset.shape[1] > 4:
        with h5py.File(filename, "a") as handle:
            handle["molecule_set_data"]["datatable"]["CHANNEL"] = np.rint(dataset[:, 4]).astype(np.int32)
    _save_extra_data(filename, extra_dict)
    return filename


def _normalize_bounds(bounds_nm):
    bounds = np.asarray(bounds_nm, dtype=float)
    if bounds.shape == (2, 2):
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
    elif bounds.size == 4:
        x_min, x_max, y_min, y_max = bounds.ravel()
    else:
        raise ValueError("Bounds must be [x_min, x_max, y_min, y_max] or [[x_min, x_max], [y_min, y_max]].")
    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    return x_min, x_max, y_min, y_max


def _dataset_bounds(dataset, padding_nm=0):
    dataset = _validate_dataset(dataset, copy=False)
    x_min, x_max = np.nanmin(dataset[:, 0]), np.nanmax(dataset[:, 0])
    y_min, y_max = np.nanmin(dataset[:, 1]), np.nanmax(dataset[:, 1])
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    return x_min - padding_nm, x_max + padding_nm, y_min - padding_nm, y_max + padding_nm


def _common_bounds(datasets, padding_nm=0):
    bounds = np.array([_dataset_bounds(dataset, padding_nm=0) for dataset in datasets], dtype=float)
    return (
        float(bounds[:, 0].min() - padding_nm),
        float(bounds[:, 1].max() + padding_nm),
        float(bounds[:, 2].min() - padding_nm),
        float(bounds[:, 3].max() + padding_nm),
    )


def _pixel_size_for_bounds(bounds_nm, pixel_size_nm, max_pixels=25_000_000):
    x_min, x_max, y_min, y_max = _normalize_bounds(bounds_nm)
    width = max(2, int(np.ceil((x_max - x_min) / pixel_size_nm)) + 1)
    height = max(2, int(np.ceil((y_max - y_min) / pixel_size_nm)) + 1)
    if width * height <= max_pixels:
        return float(pixel_size_nm)
    scale = np.sqrt(width * height / max_pixels)
    return float(pixel_size_nm * scale)


def _render_dataset_2d(dataset, render_sigma_nm=100, pixel_size_nm=50, bounds_nm=None, max_pixels=25_000_000):
    dataset = _validate_dataset(dataset, copy=False)
    if bounds_nm is None:
        bounds_nm = _dataset_bounds(dataset, padding_nm=3 * render_sigma_nm)
    x_min, x_max, y_min, y_max = _normalize_bounds(bounds_nm)
    pixel_size_nm = _pixel_size_for_bounds((x_min, x_max, y_min, y_max), pixel_size_nm, max_pixels=max_pixels)

    width = max(2, int(np.ceil((x_max - x_min) / pixel_size_nm)) + 1)
    height = max(2, int(np.ceil((y_max - y_min) / pixel_size_nm)) + 1)
    x_edges = x_min + np.arange(width + 1) * pixel_size_nm
    y_edges = y_min + np.arange(height + 1) * pixel_size_nm
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])

    in_bounds = (
        (dataset[:, 0] >= x_edges[0])
        & (dataset[:, 0] <= x_edges[-1])
        & (dataset[:, 1] >= y_edges[0])
        & (dataset[:, 1] <= y_edges[-1])
    )
    if not np.any(in_bounds):
        return np.zeros((height, width), dtype=np.float32), extent, pixel_size_nm

    image, _, _ = np.histogram2d(
        dataset[in_bounds, 1],
        dataset[in_bounds, 0],
        bins=(y_edges, x_edges),
    )
    sigma_px = render_sigma_nm / pixel_size_nm
    if sigma_px > 0:
        image = gaussian_filter(image.astype(np.float32), sigma=sigma_px, mode="nearest")
    return image.astype(np.float32), extent, pixel_size_nm


def _linear_and_log_display_settings(image, linear_percentile=99.5, log_percentiles=(5, 99.5)):
    positive = image[image > 0]
    if len(positive) == 0:
        return {"vmin": 0, "vmax": None}, None

    linear_vmax = np.percentile(positive, linear_percentile)
    log_vmin = np.percentile(positive, log_percentiles[0])
    log_vmax = np.percentile(positive, log_percentiles[1])
    if log_vmax <= log_vmin:
        log_vmax = positive.max()
    if log_vmax <= 0:
        log_norm = None
    else:
        log_norm = LogNorm(vmin=max(log_vmin, 1e-6), vmax=max(log_vmax, 1e-6))
    return {"vmin": 0, "vmax": linear_vmax}, log_norm


def _imshow_linear_and_log(axes, image, extent, cmap="hot", linear_percentile=99.5, log_percentiles=(5, 99.5)):
    linear_kwargs, log_norm = _linear_and_log_display_settings(
        image,
        linear_percentile=linear_percentile,
        log_percentiles=log_percentiles,
    )
    axes[0].imshow(image, origin="lower", extent=extent, cmap=cmap, **linear_kwargs)
    axes[0].set_title(f"Linear, clipped p{linear_percentile:g}")
    axes[1].imshow(image + 1e-6, origin="lower", extent=extent, cmap=cmap, norm=log_norm)
    axes[1].set_title("Log view")
    for ax in axes:
        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")


def _crop_dataset(dataset, bounds_nm):
    dataset = _validate_dataset(dataset)
    x_min, x_max, y_min, y_max = _normalize_bounds(bounds_nm)
    keep = (
        (dataset[:, 0] >= x_min)
        & (dataset[:, 0] <= x_max)
        & (dataset[:, 1] >= y_min)
        & (dataset[:, 1] <= y_max)
    )
    return dataset[keep].copy()


def _compact_frame_axis(dataset):
    dataset = _validate_dataset(dataset)
    original_frames = dataset[:, 3].astype(np.int64)
    unique_frames = np.unique(original_frames)
    compact_lookup = {frame: idx for idx, frame in enumerate(unique_frames)}
    compact_frames = np.fromiter((compact_lookup[frame] for frame in original_frames), dtype=np.int64)
    compacted = dataset.copy()
    compacted[:, 3] = compact_frames
    return compacted, {
        "original_frames": unique_frames,
        "compacted_frames": np.arange(len(unique_frames), dtype=np.int64),
    }


def _split_frame_blocks(frames, frames_per_timepoint=None, n_timepoints=None):
    unique_frames = np.unique(frames.astype(np.int64))
    if len(unique_frames) == 0:
        return []

    if frames_per_timepoint is not None:
        if int(frames_per_timepoint) < 1:
            raise ValueError("frames_per_timepoint must be >= 1.")
        frame_min = int(unique_frames.min())
        frame_max = int(unique_frames.max())
        blocks = []
        for start in range(frame_min, frame_max + 1, int(frames_per_timepoint)):
            stop = start + int(frames_per_timepoint)
            block = unique_frames[(unique_frames >= start) & (unique_frames < stop)]
            if len(block):
                blocks.append(block)
        return blocks

    if n_timepoints is not None:
        if int(n_timepoints) < 1:
            raise ValueError("n_timepoints must be >= 1.")
        return [block for block in np.array_split(unique_frames, int(n_timepoints)) if len(block)]

    gap_indices = np.where(np.diff(unique_frames) > 1)[0] + 1
    return [block for block in np.split(unique_frames, gap_indices) if len(block)]


def _as_int_set(values):
    if values is None:
        return set()
    if np.isscalar(values):
        return {int(values)}
    return {int(value) for value in values}


def _format_array_literal(name, array, precision=1):
    body = np.array2string(
        np.asarray(array),
        precision=precision,
        separator=", ",
        suppress_small=False,
    )
    return f"{name}=np.array({body})"


def _count_locs_near_centers(dataset, centers_nm, radius_nm):
    dataset = _validate_dataset(dataset, copy=False)
    centers = np.asarray(centers_nm, dtype=float).reshape(-1, 2)
    if len(centers) == 0:
        return np.empty(0, dtype=np.int64)
    tree = cKDTree(dataset[:, :2])
    return np.asarray([len(tree.query_ball_point(center, r=radius_nm)) for center in centers], dtype=np.int64)


def _detect_bead_centers(
    dataset,
    render_sigma_nm=150,
    pixel_size_nm=100,
    percentile=99.5,
    min_distance_nm=1_000,
    max_beads=32,
    min_component_pixels=1,
    max_component_pixels=80,
    max_aspect_ratio=2.0,
    count_radius_nm=500,
    min_locs=None,
    return_diagnostics=False,
):
    dataset = _validate_dataset(dataset, copy=False)
    image, extent, actual_pixel_size_nm = _render_dataset_2d(
        dataset,
        render_sigma_nm=render_sigma_nm,
        pixel_size_nm=pixel_size_nm,
    )
    positive = image[image > 0]
    if len(positive) == 0:
        centers = np.empty((0, 2), dtype=float)
        if return_diagnostics:
            return centers, {"selected": [], "rejected": [], "threshold": np.nan}
        return centers

    threshold = np.percentile(positive, percentile)
    x0, _, y0, _ = extent

    candidate_mask = image >= threshold
    component_labels, n_components = label(candidate_mask)
    candidates = []
    rejected = []
    for component_id in range(1, n_components + 1):
        y_idx, x_idx = np.nonzero(component_labels == component_id)
        component_pixels = len(x_idx)
        if component_pixels < min_component_pixels:
            continue
        weights = image[y_idx, x_idx]
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        center_x_px = np.sum((x_idx + 0.5) * weights) / weight_sum
        center_y_px = np.sum((y_idx + 0.5) * weights) / weight_sum
        xy_centered = np.column_stack((x_idx + 0.5 - center_x_px, y_idx + 0.5 - center_y_px))
        if component_pixels > 1:
            covariance = (xy_centered * weights[:, np.newaxis]).T @ xy_centered / weight_sum
            eigenvalues = np.linalg.eigvalsh(covariance)
            aspect_ratio = float(np.sqrt((eigenvalues[-1] + 1e-12) / (eigenvalues[0] + 1e-12)))
        else:
            aspect_ratio = 1.0
        bbox_width = int(x_idx.max() - x_idx.min() + 1)
        bbox_height = int(y_idx.max() - y_idx.min() + 1)
        bbox_aspect_ratio = max(bbox_width / max(bbox_height, 1), bbox_height / max(bbox_width, 1))
        peak_value = weights.max()
        integrated_value = weight_sum
        center_nm = np.array(
            [
                x0 + center_x_px * actual_pixel_size_nm,
                y0 + center_y_px * actual_pixel_size_nm,
            ],
            dtype=float,
        )
        diagnostic = {
            "center_nm": center_nm,
            "component_pixels": component_pixels,
            "aspect_ratio": aspect_ratio,
            "bbox_aspect_ratio": bbox_aspect_ratio,
            "peak_value": peak_value,
            "integrated_value": integrated_value,
        }
        if max_component_pixels is not None and component_pixels > max_component_pixels:
            diagnostic_rejected = diagnostic.copy()
            diagnostic_rejected["reason"] = "too_large"
            rejected.append(diagnostic_rejected)
            continue
        if max_aspect_ratio is not None and max(aspect_ratio, bbox_aspect_ratio) > max_aspect_ratio:
            diagnostic_rejected = diagnostic.copy()
            diagnostic_rejected["reason"] = "elongated"
            rejected.append(diagnostic_rejected)
            continue
        candidates.append((peak_value, integrated_value, center_x_px, center_y_px, diagnostic))

    if not candidates and not rejected:
        window_px = max(3, int(np.ceil(min_distance_nm / actual_pixel_size_nm)))
        local_max = maximum_filter(image, size=window_px, mode="nearest")
        peak_mask = (image == local_max) & (image >= threshold)
        peak_y, peak_x = np.nonzero(peak_mask)
        values = image[peak_y, peak_x]
        candidates = [
            (
                value,
                value,
                x + 0.5,
                y + 0.5,
                {
                    "center_nm": np.array([x0 + (x + 0.5) * actual_pixel_size_nm,
                                           y0 + (y + 0.5) * actual_pixel_size_nm]),
                    "component_pixels": 1,
                    "aspect_ratio": 1.0,
                    "bbox_aspect_ratio": 1.0,
                    "peak_value": value,
                    "integrated_value": value,
                },
            )
            for value, x, y in zip(values, peak_x, peak_y)
        ]

    tree = cKDTree(dataset[:, :2])
    counted_candidates = []
    for peak_value, integrated_value, center_x_px, center_y_px, diagnostic in candidates:
        center_nm = diagnostic["center_nm"]
        loc_count = len(tree.query_ball_point(center_nm, r=count_radius_nm))
        diagnostic["loc_count"] = loc_count
        if min_locs is not None and loc_count < min_locs:
            diagnostic_rejected = diagnostic.copy()
            diagnostic_rejected["reason"] = "too_few_locs"
            rejected.append(diagnostic_rejected)
            continue
        counted_candidates.append((loc_count, peak_value, integrated_value, center_x_px, center_y_px, diagnostic))
    candidates = counted_candidates

    selected = []
    selected_diagnostics = []
    for _loc_count, _peak_value, _integrated_value, center_x_px, center_y_px, diagnostic in sorted(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    ):
        center = np.array(
            [
                x0 + center_x_px * actual_pixel_size_nm,
                y0 + center_y_px * actual_pixel_size_nm,
            ],
            dtype=float,
        )
        if all(np.linalg.norm(center - other) >= min_distance_nm for other in selected):
            selected.append(center)
            selected_diagnostics.append(diagnostic)
        if len(selected) >= max_beads:
            break

    if not selected:
        centers = np.empty((0, 2), dtype=float)
    else:
        centers = np.vstack(selected)
    if return_diagnostics:
        return centers, {"selected": selected_diagnostics, "rejected": rejected, "threshold": threshold}
    return centers


def _review_bead_centers_interactively(
    dataset,
    bead_centers_nm,
    bead_diagnostics=None,
    bead_radius_nm=1000,
    bead_count_radius_nm=500,
    render_sigma_nm=100,
    pixel_size_nm=50,
    print_precision=1,
):
    dataset = _validate_dataset(dataset, copy=False)
    centers = [np.asarray(center, dtype=float) for center in np.asarray(bead_centers_nm).reshape(-1, 2)]
    bead_diagnostics = bead_diagnostics or {"rejected": []}
    rejected_centers = np.asarray([item["center_nm"] for item in bead_diagnostics.get("rejected", [])])

    bounds = _dataset_bounds(dataset, padding_nm=300)
    image_before, extent, _ = _render_dataset_2d(
        dataset,
        render_sigma_nm=render_sigma_nm,
        pixel_size_nm=pixel_size_nm,
        bounds_nm=bounds,
    )

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.12)
    image_axes = tuple(axes.flat)

    def centers_array():
        if not centers:
            return np.empty((0, 2), dtype=float)
        return np.vstack(centers)

    def redraw():
        current_centers = centers_array()
        bead_mask = _points_near_centers(dataset, current_centers, bead_radius_nm)
        dataset_after = dataset[~bead_mask].copy()
        if len(dataset_after):
            image_after, _, _ = _render_dataset_2d(
                dataset_after,
                render_sigma_nm=render_sigma_nm,
                pixel_size_nm=pixel_size_nm,
                bounds_nm=bounds,
            )
        else:
            image_after = np.zeros_like(image_before)

        for ax in image_axes:
            ax.clear()
        _imshow_linear_and_log(axes[0], image_before, extent, cmap="hot")
        _imshow_linear_and_log(axes[1], image_after, extent, cmap="hot")
        axes[0, 0].set_ylabel("Before removal\ny [nm]")
        axes[1, 0].set_ylabel("After removal\ny [nm]")

        for ax in image_axes:
            if len(current_centers):
                ax.scatter(
                    current_centers[:, 0],
                    current_centers[:, 1],
                    facecolors="none",
                    edgecolors="tab:red",
                    s=80,
                    linewidths=1.5,
                )
            if len(rejected_centers):
                ax.scatter(rejected_centers[:, 0], rejected_centers[:, 1], marker="x", color="tab:blue", alpha=0.5)

        fig.suptitle("Review bead centers: left-click adds, right-click removes nearest, then Accept")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes not in image_axes or event.xdata is None or event.ydata is None:
            return
        clicked = np.array([event.xdata, event.ydata], dtype=float)
        if event.button == 1:
            centers.append(clicked)
        elif event.button == 3 and centers:
            current_centers = centers_array()
            nearest = int(np.argmin(np.linalg.norm(current_centers - clicked, axis=1)))
            centers.pop(nearest)
        redraw()

    button_ax = fig.add_axes((0.82, 0.025, 0.14, 0.055))
    accept_button = Button(button_ax, "Accept")

    def on_accept(_event):
        plt.close(fig)

    click_connection = fig.canvas.mpl_connect("button_press_event", on_click)
    accept_button.on_clicked(on_accept)
    redraw()
    plt.show()
    fig.canvas.mpl_disconnect(click_connection)

    final_centers = centers_array()
    print("Paste this to reuse these bead centers:")
    print(_format_array_literal("bead_centers_nm", final_centers, precision=print_precision))
    if len(final_centers):
        counts = _count_locs_near_centers(dataset, final_centers, bead_count_radius_nm)
        print("Selected bead localizations within count radius:")
        for idx, (center, count) in enumerate(zip(final_centers, counts)):
            print(f"  {idx}: x={center[0]:.1f}, y={center[1]:.1f}, loc_count={int(count)}")

    return final_centers


def _points_near_centers(dataset, centers_nm, radius_nm):
    dataset = _validate_dataset(dataset, copy=False)
    centers = np.asarray(centers_nm, dtype=float).reshape(-1, 2)
    if len(centers) == 0 or radius_nm is None or radius_nm <= 0:
        return np.zeros(len(dataset), dtype=bool)
    near = np.zeros(len(dataset), dtype=bool)
    radius_sq = float(radius_nm) ** 2
    xy = dataset[:, :2]
    for center in centers:
        near |= np.sum((xy - center) ** 2, axis=1) <= radius_sq
    return near


def _save_drift_h5(filename, drift_compact_frames, drift_original_frames=None, extra_dict=None):
    if filename is None:
        return None
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as handle:
        handle["drift_compact_frames_nm"] = np.asarray(drift_compact_frames)
        if drift_original_frames is not None:
            handle["drift_original_frames_nm"] = np.asarray(drift_original_frames)
        if extra_dict:
            group = handle.create_group("extra_data")
            for key, value in extra_dict.items():
                _write_h5_value(group, str(key), value)
    return filename


def _build_final_drift_dataframe(drift_curves, rcc_shifts_nm, split_metadata, frame_mapping=None):
    rows = []
    timepoint_labels = split_metadata.get("timepoint_labels", np.arange(len(drift_curves)))
    continuous_frame_offset = 0

    for timepoint_index, drift_curve in enumerate(drift_curves):
        drift_curve = np.asarray(drift_curve, dtype=float)
        local_frames = drift_curve[:, 3].astype(np.int64)
        original_frames = np.asarray(split_metadata["original_frames_per_timepoint"][timepoint_index], dtype=np.int64)
        valid = (local_frames >= 0) & (local_frames < len(original_frames))
        if not np.all(valid):
            drift_curve = drift_curve[valid]
            local_frames = local_frames[valid]

        source_frames = original_frames[local_frames]
        if frame_mapping is not None:
            source_frames = frame_mapping["original_frames"][source_frames.astype(np.int64)]

        rcc_shift = np.asarray(rcc_shifts_nm[timepoint_index], dtype=float)
        comet_drift = drift_curve[:, :3]
        final_drift_to_subtract = comet_drift - rcc_shift[np.newaxis, :]
        timepoint_label = int(timepoint_labels[timepoint_index])

        for row_idx, local_frame in enumerate(local_frames):
            rows.append(
                {
                    "original_frame": int(source_frames[row_idx]),
                    "continuous_frame": int(continuous_frame_offset + local_frame),
                    "timepoint_index": int(timepoint_index),
                    "timepoint_label": timepoint_label,
                    "local_frame": int(local_frame),
                    "dx_nm": float(final_drift_to_subtract[row_idx, 0]),
                    "dy_nm": float(final_drift_to_subtract[row_idx, 1]),
                    "dz_nm": float(final_drift_to_subtract[row_idx, 2]),
                    "comet_dx_nm": float(comet_drift[row_idx, 0]),
                    "comet_dy_nm": float(comet_drift[row_idx, 1]),
                    "comet_dz_nm": float(comet_drift[row_idx, 2]),
                    "rcc_dx_nm": float(rcc_shift[0]),
                    "rcc_dy_nm": float(rcc_shift[1]),
                    "rcc_dz_nm": float(rcc_shift[2]),
                }
            )
        if len(local_frames):
            continuous_frame_offset += int(local_frames.max()) + 1

    columns = [
        "original_frame",
        "continuous_frame",
        "timepoint_index",
        "timepoint_label",
        "local_frame",
        "dx_nm",
        "dy_nm",
        "dz_nm",
        "comet_dx_nm",
        "comet_dy_nm",
        "comet_dz_nm",
        "rcc_dx_nm",
        "rcc_dy_nm",
        "rcc_dz_nm",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["timepoint_index", "original_frame"]).reset_index(drop=True)


def _save_final_drift_csv(final_drift_df, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    final_drift_df.to_csv(filename, index=False, float_format="%.6f")
    return filename


def _plot_final_drift(final_drift_df):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    components = [("dx_nm", "x"), ("dy_nm", "y"), ("dz_nm", "z")]
    for ax, (column, label) in zip(axes, components):
        for timepoint_label, group in final_drift_df.groupby("timepoint_label"):
            group = group.sort_values("original_frame")
            ax.plot(group["original_frame"], group[column], label=f"tp {timepoint_label}")
        ax.set_ylabel(f"{label} drift [nm]")
    axes[-1].set_xlabel("Original frame")
    axes[0].set_title("Final drift estimate to subtract (COMET - RCC)")
    axes[0].legend()
    plt.tight_layout()
    plt.show()


def _plot_final_drift_continuous(final_drift_df):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    components = [("dx_nm", "x"), ("dy_nm", "y"), ("dz_nm", "z")]
    plot_df = final_drift_df.sort_values("continuous_frame")
    for ax, (column, label) in zip(axes, components):
        ax.plot(plot_df["continuous_frame"], plot_df[column], color="tab:blue", lw=1.2)
        ax.set_ylabel(f"{label} drift [nm]")

    boundaries = (
        plot_df.groupby("timepoint_index")["continuous_frame"]
        .min()
        .sort_index()
        .to_numpy()
    )
    labels = (
        plot_df.groupby("timepoint_index")["timepoint_label"]
        .first()
        .sort_index()
        .to_numpy()
    )
    for boundary, label in zip(boundaries, labels):
        for ax in axes:
            ax.axvline(boundary, color="0.75", lw=0.8, ls="--")
        axes[0].text(boundary, axes[0].get_ylim()[1], f" tp {label}", va="top", ha="left", fontsize=8)

    axes[-1].set_xlabel("Continuous retained frame")
    axes[0].set_title("Final drift estimate on compact retained-frame axis")
    plt.tight_layout()
    plt.show()


def _phase_correlation_shift(reference, moving):
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    reference = reference - np.mean(reference)
    moving = moving - np.mean(moving)
    cross_power = np.fft.fft2(reference) * np.fft.fft2(moving).conj()
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.abs(np.fft.ifft2(cross_power))
    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape), dtype=float)
    shape = np.array(correlation.shape, dtype=float)

    for axis in range(2):
        idx = int(peak[axis])
        prev_idx = (idx - 1) % correlation.shape[axis]
        next_idx = (idx + 1) % correlation.shape[axis]
        if axis == 0:
            before = correlation[prev_idx, int(peak[1])]
            center = correlation[idx, int(peak[1])]
            after = correlation[next_idx, int(peak[1])]
        else:
            before = correlation[int(peak[0]), prev_idx]
            center = correlation[int(peak[0]), idx]
            after = correlation[int(peak[0]), next_idx]
        denominator = before - 2 * center + after
        if abs(denominator) > 1e-12:
            peak[axis] += 0.5 * (before - after) / denominator

    midpoints = np.fix(shape / 2)
    peak[peak > midpoints] -= shape[peak > midpoints]
    return peak


def render_and_crop_out_single_cell(
    dataset,
    render_sigma_nm=100,
    pixel_size_nm=50,
    crop_bounds_nm=None,
    save_path=None,
    pixelsize_nm=160,
    show=True,
    return_bounds=False,
    display_percentile=99.5,
    log_display_percentiles=(5, 99.5),
    crop_print_precision=1,
):
    """
    Render a 2D preview and crop one cell with either explicit bounds or an interactive rectangle.

    Parameters
    ----------
    dataset : ndarray or path
        Localization data with columns [x_nm, y_nm, z_nm, frame].
    crop_bounds_nm : sequence or None
        [x_min, x_max, y_min, y_max]. If omitted, an interactive Matplotlib selector is shown.
    save_path : str or Path or None
        Optional HDF5 output path for the cropped molecule set.
    return_bounds : bool
        If True, return (cropped_dataset, crop_bounds_nm).
    """
    dataset = _validate_dataset(dataset)
    if crop_bounds_nm is None:
        if not show:
            raise ValueError("crop_bounds_nm must be provided when show=False.")

        image, extent, _ = _render_dataset_2d(dataset, render_sigma_nm, pixel_size_nm)
        selected = {"bounds": None}

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.15)
        _imshow_linear_and_log(
            axes,
            image,
            extent,
            cmap="hot",
            linear_percentile=display_percentile,
            log_percentiles=log_display_percentiles,
        )
        fig.suptitle("Select one cell in either panel, then click Accept")

        def on_select(click, release):
            if click.xdata is None or click.ydata is None or release.xdata is None or release.ydata is None:
                return
            selected["bounds"] = _normalize_bounds((click.xdata, release.xdata, click.ydata, release.ydata))

        selectors = []
        for ax in axes:
            selector = RectangleSelector(
                ax,
                on_select,
                useblit=True,
                button=[1],
                minspanx=pixel_size_nm,
                minspany=pixel_size_nm,
                spancoords="data",
                interactive=True,
            )
            selector.set_active(True)
            selectors.append(selector)

        button_ax = fig.add_axes((0.8, 0.025, 0.15, 0.06))
        accept_button = Button(button_ax, "Accept")

        def on_accept(_event):
            plt.close(fig)

        accept_button.on_clicked(on_accept)
        plt.show()

        if selected["bounds"] is None:
            raise RuntimeError("No crop rectangle was selected.")
        crop_bounds_nm = selected["bounds"]
        rounded_bounds = tuple(round(float(value), crop_print_precision) for value in crop_bounds_nm)
        print(f"Paste this next time: crop_bounds_nm={rounded_bounds}")
    else:
        crop_bounds_nm = _normalize_bounds(crop_bounds_nm)

    cropped = _crop_dataset(dataset, crop_bounds_nm)
    if save_path is not None:
        _save_dataset_h5(
            cropped,
            save_path,
            pixelsize_nm=pixelsize_nm,
            extra_dict={"crop_bounds_nm": np.asarray(crop_bounds_nm, dtype=float)},
        )

    if return_bounds:
        return cropped, crop_bounds_nm
    return cropped


def remove_frames_from_z_planes_with_neglectable_content(
    dataset,
    threshold,
    frame_pack_size=1,
    sanity_check=False,
    save_path=None,
    pixelsize_nm=160,
):
    """
    Remove entire frames, or frame packs, with too few localizations and compact the frame axis.

    ``threshold`` is interpreted as an absolute localization count when >= 1 and as a
    fraction of the maximum per-pack count when 0 < threshold < 1.

    Returns
    -------
    cleaned_dataset, mapping
        mapping stores the removed frames and the compacted-frame to original-frame transform.
    """
    dataset = _validate_dataset(dataset)
    if int(frame_pack_size) < 1:
        raise ValueError("frame_pack_size must be >= 1.")
    frame_pack_size = int(frame_pack_size)

    frames = dataset[:, 3].astype(np.int64)
    unique_frames, counts = np.unique(frames, return_counts=True)
    frame_min = int(unique_frames.min())
    pack_start_per_frame = ((unique_frames - frame_min) // frame_pack_size) * frame_pack_size + frame_min
    pack_starts, inverse = np.unique(pack_start_per_frame, return_inverse=True)
    pack_counts = np.bincount(inverse, weights=counts).astype(np.int64)
    pack_ends = pack_starts + frame_pack_size - 1

    if 0 < threshold < 1:
        threshold_locs = float(threshold) * float(pack_counts.max())
    else:
        threshold_locs = float(threshold)

    keep_pack_starts = pack_starts[pack_counts > threshold_locs]
    removed_pack_starts = pack_starts[pack_counts <= threshold_locs]
    keep_frames = unique_frames[np.isin(pack_start_per_frame, keep_pack_starts)]
    removed_frames = unique_frames[np.isin(pack_start_per_frame, removed_pack_starts)]
    if len(keep_frames) == 0:
        raise ValueError("Frame-pack filtering removed every frame. Lower the threshold.")

    keep_lookup = {frame: idx for idx, frame in enumerate(keep_frames)}
    keep_mask = np.isin(frames, keep_frames)
    cleaned = dataset[keep_mask].copy()
    cleaned[:, 3] = np.fromiter((keep_lookup[int(frame)] for frame in cleaned[:, 3]), dtype=np.int64)

    mapping = {
        "threshold_locs": threshold_locs,
        "frame_pack_size": frame_pack_size,
        "frame_count_original_frames": unique_frames,
        "frame_count_locs": counts,
        "pack_start_frames": pack_starts,
        "pack_end_frames": pack_ends,
        "pack_locs": pack_counts,
        "removed_pack_start_frames": removed_pack_starts,
        "removed_frames": removed_frames,
        "compacted_frames": np.arange(len(keep_frames), dtype=np.int64),
        "original_frames": keep_frames,
    }

    if sanity_check:
        fig, ax = plt.subplots()
        if frame_pack_size == 1:
            ax.plot(unique_frames, counts, lw=1)
            ax.set_xlabel("Original frame")
            ax.set_ylabel("Localizations")
        else:
            ax.plot(pack_starts, pack_counts, lw=1, marker="o", ms=3)
            ax.set_xlabel(f"Original frame pack start, width={frame_pack_size}")
            ax.set_ylabel("Localizations per pack")
        ax.axhline(threshold_locs, color="tab:red", ls="--", label="threshold")
        if len(removed_pack_starts):
            ax.scatter(
                removed_pack_starts,
                pack_counts[np.isin(pack_starts, removed_pack_starts)],
                color="tab:red",
                s=20,
            )
        ax.set_title("Removed low-content frame packs")
        ax.legend()
        plt.show()

    if save_path is not None:
        _save_dataset_h5(cleaned, save_path, pixelsize_nm=pixelsize_nm, extra_dict=mapping)

    return cleaned, mapping


def crop_beads_and_split_timepoints(
    dataset,
    frames_per_timepoint=None,
    n_timepoints=None,
    output_dir=None,
    pixelsize_nm=160,
    split_by_channel=None,
    channel_column=4,
    exclude_timepoints=None,
    bead_centers_nm=None,
    detect_beads=True,
    remove_beads=False,
    bead_radius_nm=1000,
    bead_detection_percentile=99.9,
    bead_min_distance_nm=1_000,
    bead_max_beads=32,
    bead_min_component_pixels=1,
    bead_max_component_pixels=80,
    bead_max_aspect_ratio=2.0,
    bead_count_radius_nm=500,
    bead_min_locs=None,
    sanity_check=False,
):
    """
    Optionally detect bead-like dense spots, remove them, and split into timepoints.

    When a fifth column is present, it is treated as the molecule-set CHANNEL by
    default and used as the timepoint label. ``exclude_timepoints`` then contains
    channel labels to skip, e.g. ``exclude_timepoints=(2,)``. For frame-based
    splitting, ``exclude_timepoints`` contains output timepoint indices to skip.
    """
    dataset = _validate_dataset(dataset)
    bead_diagnostics = {"selected": [], "rejected": [], "threshold": np.nan}
    exclude_timepoints = _as_int_set(exclude_timepoints)
    if split_by_channel is None:
        split_by_channel = (
            dataset.shape[1] > channel_column
            and frames_per_timepoint is None
            and n_timepoints is None
        )

    if bead_centers_nm is None and detect_beads:
        bead_centers_nm, bead_diagnostics = _detect_bead_centers(
            dataset,
            percentile=bead_detection_percentile,
            min_distance_nm=bead_min_distance_nm,
            max_beads=bead_max_beads,
            min_component_pixels=bead_min_component_pixels,
            max_component_pixels=bead_max_component_pixels,
            max_aspect_ratio=bead_max_aspect_ratio,
            count_radius_nm=bead_count_radius_nm,
            min_locs=bead_min_locs,
            return_diagnostics=True,
        )
    elif bead_centers_nm is None:
        bead_centers_nm = np.empty((0, 2), dtype=float)
    else:
        bead_centers_nm = np.asarray(bead_centers_nm, dtype=float).reshape(-1, 2)

    if sanity_check:
        bead_centers_nm = _review_bead_centers_interactively(
            dataset,
            bead_centers_nm,
            bead_diagnostics=bead_diagnostics,
            bead_radius_nm=bead_radius_nm,
            bead_count_radius_nm=bead_count_radius_nm,
        )

    bead_mask = _points_near_centers(dataset, bead_centers_nm, bead_radius_nm)
    split_source = dataset[~bead_mask].copy() if remove_beads else dataset.copy()

    timepoints = []
    original_frames_per_timepoint = []
    timepoint_labels = []
    if split_by_channel:
        if split_source.shape[1] <= channel_column:
            raise ValueError("split_by_channel=True requires a CHANNEL column.")
        channel_labels = np.rint(split_source[:, channel_column]).astype(np.int64)
        split_blocks = [
            (int(channel), channel_labels == channel)
            for channel in np.unique(channel_labels)
            if int(channel) not in exclude_timepoints
        ]
    else:
        frame_blocks = _split_frame_blocks(
            split_source[:, 3].astype(np.int64),
            frames_per_timepoint=frames_per_timepoint,
            n_timepoints=n_timepoints,
        )
        split_blocks = [
            (idx, np.isin(split_source[:, 3].astype(np.int64), block))
            for idx, block in enumerate(frame_blocks)
            if idx not in exclude_timepoints
        ]

    for idx, (timepoint_label, mask) in enumerate(split_blocks):
        timepoint = split_source[mask].copy()
        compacted, frame_mapping = _compact_frame_axis(timepoint)
        timepoints.append(compacted)
        original_frames_per_timepoint.append(frame_mapping["original_frames"])
        timepoint_labels.append(timepoint_label)

        if output_dir is not None:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            _save_dataset_h5(
                compacted,
                output_dir_path / f"timepoint_{idx:03d}_label_{timepoint_label}.h5",
                pixelsize_nm=pixelsize_nm,
                extra_dict={
                    "timepoint_index": idx,
                    "timepoint_label": timepoint_label,
                    "original_frames": frame_mapping["original_frames"],
                    "compacted_frames": frame_mapping["compacted_frames"],
                    "split_by_channel": bool(split_by_channel),
                    "bead_centers_nm": bead_centers_nm,
                    "bead_radius_nm": bead_radius_nm,
                    "beads_removed": bool(remove_beads),
                },
            )

    metadata = {
        "bead_centers_nm": bead_centers_nm,
        "bead_radius_nm": bead_radius_nm,
        "bead_count_radius_nm": bead_count_radius_nm,
        "bead_min_locs": -1 if bead_min_locs is None else int(bead_min_locs),
        "bead_selected_loc_counts": _count_locs_near_centers(dataset, bead_centers_nm, bead_count_radius_nm),
        "beads_removed": bool(remove_beads),
        "n_bead_localizations": int(bead_mask.sum()),
        "original_frames_per_timepoint": original_frames_per_timepoint,
        "timepoint_sizes": np.asarray([len(tp) for tp in timepoints], dtype=np.int64),
        "timepoint_labels": np.asarray(timepoint_labels, dtype=np.int64),
        "split_by_channel": bool(split_by_channel),
        "excluded_timepoints": np.asarray(sorted(exclude_timepoints), dtype=np.int64),
    }

    return timepoints, metadata


def comet_correct_single_timepoint(
    dataset,
    n_locs_per_segment=500,
    gt_drift=None,
    sanity_check=False,
    save_path=None,
    drift_save_path=None,
    pixelsize_nm=160,
    max_drift_nm=300,
    initial_sigma_nm=None,
    target_sigma_nm=10,
    max_locs_per_segment=None,
    boxcar_width=1,
    drift_max_bound_factor=2,
    interpolation_method="cubic",
    mode="cuda",
):
    """
    Run COMET on one timepoint and preserve the input frame labels in the corrected output.
    """
    dataset = _validate_dataset(dataset)
    work, frame_mapping = _compact_frame_axis(dataset)
    original_frame_labels = dataset[:, 3].copy()

    comet_run = _get_comet_run_kd()
    drift_compact, corrected = comet_run(
        dataset=work[:, :4].copy(),
        segmentation_mode=1,
        segmentation_var=int(n_locs_per_segment),
        initial_sigma_nm=initial_sigma_nm,
        gt_drift=gt_drift,
        display=False,
        return_corrected_locs=True,
        max_drift_nm=max_drift_nm,
        target_sigma_nm=target_sigma_nm,
        boxcar_width=boxcar_width,
        drift_max_bound_factor=drift_max_bound_factor,
        max_locs_per_segment=max_locs_per_segment,
        interpolation_method=interpolation_method,
        mode=mode,
    )
    corrected[:, 3] = original_frame_labels

    drift_original = drift_compact.copy()
    drift_frames = drift_original[:, 3].astype(np.int64)
    valid = drift_frames < len(frame_mapping["original_frames"])
    drift_original[valid, 3] = frame_mapping["original_frames"][drift_frames[valid]]

    extra = {
        "n_locs_per_segment": int(n_locs_per_segment),
        "max_drift_nm": max_drift_nm,
        "initial_sigma_nm": -1 if initial_sigma_nm is None else initial_sigma_nm,
        "target_sigma_nm": target_sigma_nm,
        "boxcar_width": boxcar_width,
        "drift_max_bound_factor": drift_max_bound_factor,
        "interpolation_method": interpolation_method,
        "mode": mode,
        "input_original_frames": frame_mapping["original_frames"],
        "input_compacted_frames": frame_mapping["compacted_frames"],
    }

    if save_path is not None:
        _save_dataset_h5(corrected, save_path, pixelsize_nm=pixelsize_nm, extra_dict=extra)
    if drift_save_path is not None:
        _save_drift_h5(drift_save_path, drift_compact, drift_original, extra_dict=extra)

    if sanity_check:
        fig, ax = plt.subplots()
        ax.plot(drift_original[:, 3], drift_original[:, 0], label="x")
        ax.plot(drift_original[:, 3], drift_original[:, 1], label="y")
        ax.plot(drift_original[:, 3], drift_original[:, 2], label="z")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Drift [nm]")
        ax.legend()
        ax.set_title("COMET drift estimate")
        plt.show()

    return corrected, drift_original


def align_comet_corrected_timepoints_with_rcc(
    timepoint_datasets=None,
    filepaths=None,
    output_dir=None,
    pixelsize_nm=160,
    render_sigma_nm=100,
    pixel_size_nm=50,
    reference_index=0,
    sanity_check=False,
):
    """
    Render COMET-corrected timepoints and align them to one reference by phase RCC.

    Returns
    -------
    aligned_timepoints, shifts_nm
        shifts_nm has columns [dx_nm, dy_nm, dz_nm] applied to each timepoint.
    """
    if timepoint_datasets is None:
        if filepaths is None:
            selected = _ask_open("Select one COMET-corrected HDF5 timepoint")
            filepaths = [selected]
        timepoint_datasets = [_validate_dataset(path) for path in filepaths]
    else:
        timepoint_datasets = [_validate_dataset(dataset) for dataset in timepoint_datasets]

    if not timepoint_datasets:
        raise ValueError("No timepoints supplied for RCC alignment.")
    if reference_index < 0 or reference_index >= len(timepoint_datasets):
        raise ValueError("reference_index is outside the timepoint list.")

    bounds = _common_bounds(timepoint_datasets, padding_nm=3 * render_sigma_nm)
    pixel_size_nm = _pixel_size_for_bounds(bounds, pixel_size_nm)
    images = [
        _render_dataset_2d(
            dataset,
            render_sigma_nm=render_sigma_nm,
            pixel_size_nm=pixel_size_nm,
            bounds_nm=bounds,
        )[0]
        for dataset in timepoint_datasets
    ]
    reference = images[reference_index]
    shifts_nm = np.zeros((len(timepoint_datasets), 3), dtype=float)
    aligned = []

    for idx, (dataset, image) in enumerate(zip(timepoint_datasets, images)):
        if idx == reference_index:
            shift_rc = np.zeros(2, dtype=float)
        else:
            shift_rc = _phase_correlation_shift(reference, image)
        dx_nm = shift_rc[1] * pixel_size_nm
        dy_nm = shift_rc[0] * pixel_size_nm
        shifts_nm[idx, :] = (dx_nm, dy_nm, 0.0)
        corrected = dataset.copy()
        corrected[:, 0] += dx_nm
        corrected[:, 1] += dy_nm
        aligned.append(corrected)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, dataset in enumerate(aligned):
            _save_dataset_h5(
                dataset,
                output_dir / f"timepoint_{idx:03d}_rcc_aligned.h5",
                pixelsize_nm=pixelsize_nm,
                extra_dict={
                    "rcc_shift_nm": shifts_nm[idx],
                    "reference_index": reference_index,
                    "render_sigma_nm": render_sigma_nm,
                    "render_pixel_size_nm": pixel_size_nm,
                },
            )
        np.savetxt(
            output_dir / "rcc_shifts_nm.csv",
            np.column_stack((np.arange(len(shifts_nm)), shifts_nm)),
            delimiter=",",
            header="timepoint,dx_nm,dy_nm,dz_nm",
            comments="",
        )

    if sanity_check:
        fig, ax = plt.subplots()
        ax.plot(shifts_nm[:, 0], label="x")
        ax.plot(shifts_nm[:, 1], label="y")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Applied RCC shift [nm]")
        ax.legend()
        ax.set_title("RCC alignment shifts")
        plt.show()

    return aligned, shifts_nm


def load_full_dataset_apply_comet_correction_and_rcc_alignment(
    input_file=None,
    output_file=None,
    output_dir=None,
    export_final_drift_csv=True,
    final_drift_csv_path=None,
    crop_bounds_nm=None,
    low_content_threshold=None,
    low_content_frame_pack_size=1,
    frames_per_timepoint=None,
    n_timepoints=None,
    split_by_channel=None,
    exclude_timepoints=None,
    pixelsize_nm=160,
    render_sigma_nm=100,
    render_pixel_size_nm=50,
    n_locs_per_segment=500,
    max_drift_nm=300,
    initial_sigma_nm=None,
    target_sigma_nm=10,
    comet_mode="cuda",
    remove_beads=False,
    bead_centers_nm=None,
    bead_radius_nm=750,
    bead_detection_percentile=99.7,
    bead_min_distance_nm=1_000,
    bead_max_beads=32,
    bead_min_component_pixels=1,
    bead_max_component_pixels=80,
    bead_max_aspect_ratio=2.0,
    bead_count_radius_nm=500,
    bead_min_locs=None,
    sanity_check=False,
):
    """
    Run the full Oligostorm -> COMET -> RCC workflow and save a final HDF5 dataset.

    The returned dictionary contains intermediate datasets, drift curves, RCC shifts,
    frame mappings, and the final output path.
    """
    dataset, source_path = _load_dataset(input_file)
    dataset = _validate_dataset(dataset)

    if output_dir is None:
        if source_path is not None:
            output_dir = source_path.with_suffix("").parent
        else:
            output_dir = Path.cwd() / "oligostorm_pipeline_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_mapping = None

    dataset, crop_bounds_nm = render_and_crop_out_single_cell(
        dataset,
        render_sigma_nm=render_sigma_nm,
        pixel_size_nm=render_pixel_size_nm,
        crop_bounds_nm=crop_bounds_nm,
        save_path=output_dir / "01_cropped_cell.h5",
        pixelsize_nm=pixelsize_nm,
        show=True,
        return_bounds=True,
    )

    if low_content_threshold is not None:
        dataset, frame_mapping = remove_frames_from_z_planes_with_neglectable_content(
            dataset,
            threshold=low_content_threshold,
            frame_pack_size=low_content_frame_pack_size,
            sanity_check=sanity_check,
            save_path=output_dir / "02_removed_low_content_frames.h5",
            pixelsize_nm=pixelsize_nm,
        )

    timepoints, split_metadata = crop_beads_and_split_timepoints(
        dataset,
        frames_per_timepoint=frames_per_timepoint,
        n_timepoints=n_timepoints,
        split_by_channel=split_by_channel,
        exclude_timepoints=exclude_timepoints,
        output_dir=output_dir / "03_timepoints",
        pixelsize_nm=pixelsize_nm,
        bead_centers_nm=bead_centers_nm,
        remove_beads=remove_beads,
        bead_radius_nm=bead_radius_nm,
        bead_detection_percentile=bead_detection_percentile,
        bead_min_distance_nm=bead_min_distance_nm,
        bead_max_beads=bead_max_beads,
        bead_min_component_pixels=bead_min_component_pixels,
        bead_max_component_pixels=bead_max_component_pixels,
        bead_max_aspect_ratio=bead_max_aspect_ratio,
        bead_count_radius_nm=bead_count_radius_nm,
        bead_min_locs=bead_min_locs,
        sanity_check=sanity_check,
    )
    if not timepoints:
        raise RuntimeError("No timepoints were produced from the input dataset.")

    corrected_timepoints = []
    drift_curves = []
    comet_dir = output_dir / "04_comet_corrected"
    comet_dir.mkdir(parents=True, exist_ok=True)
    for idx, timepoint in enumerate(timepoints):
        corrected, drift = comet_correct_single_timepoint(
            timepoint,
            n_locs_per_segment=n_locs_per_segment,
            sanity_check=False,
            save_path=comet_dir / f"timepoint_{idx:03d}_comet_corrected.h5",
            drift_save_path=comet_dir / f"timepoint_{idx:03d}_drift.h5",
            pixelsize_nm=pixelsize_nm,
            max_drift_nm=max_drift_nm,
            initial_sigma_nm=initial_sigma_nm,
            target_sigma_nm=target_sigma_nm,
            mode=comet_mode,
        )
        corrected_timepoints.append(corrected)
        drift_curves.append(drift)

    aligned_timepoints, rcc_shifts_nm = align_comet_corrected_timepoints_with_rcc(
        corrected_timepoints,
        output_dir=output_dir / "05_rcc_aligned",
        pixelsize_nm=pixelsize_nm,
        render_sigma_nm=render_sigma_nm,
        pixel_size_nm=render_pixel_size_nm,
        sanity_check=False,
    )

    final_drift_df = _build_final_drift_dataframe(
        drift_curves,
        rcc_shifts_nm,
        split_metadata,
        frame_mapping=frame_mapping,
    )
    final_drift_csv_path = (
        Path(final_drift_csv_path)
        if final_drift_csv_path is not None
        else output_dir / "06_final_drift_original_frames.csv"
    )
    if export_final_drift_csv:
        final_drift_csv_path = _save_final_drift_csv(final_drift_df, final_drift_csv_path)
        print(f"Saved final drift estimate to: {final_drift_csv_path}")
    else:
        final_drift_csv_path = None

    if sanity_check:
        _plot_final_drift(final_drift_df)
        _plot_final_drift_continuous(final_drift_df)

    final_parts = []
    final_frame_to_original_frame = []
    frame_offset = 0
    for idx, timepoint in enumerate(aligned_timepoints):
        part = timepoint.copy()
        local_frames = part[:, 3].astype(np.int64)
        original_frames = split_metadata["original_frames_per_timepoint"][idx]
        if frame_mapping is not None:
            original_frames = frame_mapping["original_frames"][original_frames.astype(np.int64)]
        part[:, 3] = local_frames + frame_offset
        final_parts.append(part)
        final_frame_to_original_frame.extend(original_frames.tolist())
        frame_offset += int(local_frames.max()) + 1 if len(local_frames) else 0

    final_dataset = np.vstack(final_parts)
    if output_file is None:
        output_file = output_dir / "06_final_comet_rcc_corrected.h5"
    _save_dataset_h5(
        final_dataset,
        output_file,
        pixelsize_nm=pixelsize_nm,
        extra_dict={
            "source_file": "" if source_path is None else str(source_path),
            "crop_bounds_nm": np.asarray(crop_bounds_nm if crop_bounds_nm is not None else [], dtype=float),
            "low_content_frame_mapping": {} if frame_mapping is None else frame_mapping,
            "rcc_shifts_nm": rcc_shifts_nm,
            "final_frame_to_original_frame": np.asarray(final_frame_to_original_frame, dtype=np.int64),
            "timepoint_sizes": split_metadata["timepoint_sizes"],
            "timepoint_labels": split_metadata["timepoint_labels"],
            "excluded_timepoints": split_metadata["excluded_timepoints"],
            "split_by_channel": split_metadata["split_by_channel"],
            "bead_centers_nm": split_metadata["bead_centers_nm"],
            "beads_removed": split_metadata["beads_removed"],
            "final_drift_csv_path": "" if final_drift_csv_path is None else str(final_drift_csv_path),
        },
    )

    return {
        "final_dataset": final_dataset,
        "output_file": Path(output_file),
        "timepoints": timepoints,
        "corrected_timepoints": corrected_timepoints,
        "aligned_timepoints": aligned_timepoints,
        "drift_curves": drift_curves,
        "final_drift": final_drift_df,
        "final_drift_csv_path": final_drift_csv_path,
        "rcc_shifts_nm": rcc_shifts_nm,
        "frame_mapping": frame_mapping,
        "split_metadata": split_metadata,
    }


if __name__ == "__main__":
    folder = r"\\192.168.1.195\storm_share\storm_disk1\STORM_data1\Optimized_drift_project\Sarah_data\M7_SVABext027" \
             r"\loc5_correction\\"
    filepath = folder + r"SVABext027_MS-rep_loc005_co16_bg50_xy20-z40_ext-thunderstorm.molecule_set.h5"
    if not filepath:
        raise ValueError("Set filepath to the molecule-set .h5 file before running this script.")

    result = load_full_dataset_apply_comet_correction_and_rcc_alignment(
        input_file=filepath,
        export_final_drift_csv=True,
        final_drift_csv_path=None,
        split_by_channel=True,
        exclude_timepoints=(2,),
        #crop_bounds_nm=(26146.8, 46663.5, 23600.0, 43769.0), # loc3  #crop_bounds_nm=(25741.1, 47856.0, 28726.5, 45526.2) #loc5
        #crop_bounds_nm=None,
        low_content_threshold=0.1,
        low_content_frame_pack_size=50,
        frames_per_timepoint=None,
        n_timepoints=None,
        n_locs_per_segment=600,
        max_drift_nm=250,
        initial_sigma_nm=None,
        target_sigma_nm=30,
        comet_mode="cuda",
        remove_beads=True,
        bead_centers_nm=None,
        bead_radius_nm=1000,
        bead_detection_percentile=99.,
        bead_min_distance_nm=1000,
        bead_min_component_pixels=1,
        bead_max_component_pixels=80,
        bead_max_beads=60,
        bead_max_aspect_ratio=2.0,
        bead_count_radius_nm=500,
        bead_min_locs=None,
        sanity_check=True,
    )
    print(f"Saved final corrected dataset to: {result['output_file']}")
