# COMET

**Cost-function Optimized Maximal Overlap Drift EsTimation**

Fast, GPU-accelerated drift correction for single-molecule localization
microscopy (SMLM) datasets. COMET achieves high spatial and temporal resolution
by maximizing spatiotemporal overlap across frames using a cost-function
optimization approach.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpufit/Comet/blob/master/Colab_notebooks/COMET.ipynb)

* **Documentation:** <https://comet.smlm.tools/>
* **Source and issues:** <https://github.com/gpufit/Comet>
* **Preprint:** <https://doi.org/10.64898/2026.03.27.714864>
* **No-install web tool:** <https://www.smlm.tools>

> This page covers the Python package. The
> [repository README](https://github.com/gpufit/Comet#readme) also documents the
> CUDA/C++ library, the IDL interface, and the web tool.

## Install

```bash
pip install comet-smlm
```

For the PyTorch backend, which is useful on machines without an NVIDIA GPU
(including Apple Silicon):

```bash
pip install "comet-smlm[torch]"
```

Then check the install:

```bash
comet_self_test
```

This simulates a dataset with a known drift, recovers it, and reports which
backends it found. It needs no GPU and takes a few seconds.

### A GPU is optional

COMET runs on three backends and picks the fastest available automatically:

| Backend | Needs                           | Notes                                        |
| ------- | ------------------------------- | -------------------------------------------- |
| `cuda`  | NVIDIA GPU + CUDA driver        | Fastest; the numba-cuda kernels              |
| `torch` | `pip install "comet-smlm[torch]"` | Uses CUDA or Apple MPS, or falls back to CPU |
| `cpu`   | nothing                         | numba-compiled; no GPU needed                |

```python
import comet

print(comet.describe_backends())   # what this machine supports
print(comet.best_backend())        # what COMET will use
```

### Python 3.6 / 3.7

These are **not supported, but not blocked either**. The intent is that pip does
not refuse the install outright, so you can try COMET on an older interpreter if
that is what you have.

`pip install comet-smlm` resolves to the frozen `1.0.x` line there, whose
dependency floors are low enough for those interpreters. Upgrade pip first
(`python -m pip install --upgrade pip`) — versions before pip 9 ignore the
metadata that makes this work.

Expect to do some work: getting `numba` and `llvmlite` built or wheel-matched on
3.6/3.7 is fiddly and platform-dependent. This path is untested in CI beyond a
best-effort job, so treat it as "try it and see" rather than a supported
configuration. Use Python 3.9+ if you have the choice.

## Quickstart

```python
from comet import comet_run_kd, load_thunderstorm_csv, save_dataset_as_thunderstorm_csv

# (N, 4) array of [x_nm, y_nm, z_nm, frame]
dataset = load_thunderstorm_csv("your_data.csv")

drift, corrected = comet_run_kd(
    dataset,
    segmentation_mode=2,      # fixed number of frames per time window
    segmentation_var=60,      # 60 frames per window
    max_drift_nm=300,         # maximum expected drift
    target_sigma_nm=10,       # stop refining at this length scale
    return_corrected_locs=True,
)

save_dataset_as_thunderstorm_csv(corrected, "corrected.csv")
```

`drift` is an `(F, 4)` array of `[dx_nm, dy_nm, dz_nm, frame]`, one row per
frame.

> **Note:** `comet_run_kd` modifies the array you pass in — it subtracts the
> estimated drift from `dataset` in place. Pass `dataset.copy()` to keep the
> original.

### Input format

CSV input is ThunderSTORM-compatible. Required headers are `"frame"`,
`"x [nm]"`, `"y [nm]"`, and optionally `"z [nm]"`; extra columns are allowed.
Molecule-set HDF5 files are read with `load_normal_molecule_set`.

## Command line

```bash
comet \
  --input  your_data.csv \
  --output corrected.csv \
  --format csv \
  --segmentation_mode 2 \
  --segmentation_var 60
```

`comet --help` lists every option. The backend defaults to the fastest one
available; override it with `--mode {cuda,torch,cpu}`.

## Segmentation modes

COMET segments data temporally before estimating drift.

| Mode | Description                             | `--segmentation_var` means |
| ---- | --------------------------------------- | -------------------------- |
| 0    | Fixed number of time windows            | Number of windows          |
| 1    | Fixed number of localizations per window| Localizations per window   |
| 2    | Fixed number of frames per window (default) | Frames per window      |
| -1   | Pre-segmented: the frame column already holds a window id per localization | (ignored) |

Choose a parameter that gives you enough windows to resolve the drift, but
enough localizations per window to constrain it.

## Automatic fiducial detection

`comet_fiducials` is a standalone package shipped alongside COMET that locates
fiducial beads from one 2D histogram binned at the maximum expected drift, and
scores candidates by the number of localization pairs they contribute across
time. It imports only NumPy and SciPy — nothing from `comet` — and COMET never
calls it; use it when you want to remove beads before correction, or to keep
only the beads as a drift reference.

```python
from comet_fiducials import find_fiducials, fiducial_mask

centers_nm, report = find_fiducials(locs, max_drift_nm=250,
                                    field_bounds_nm=camera_fov_nm)
radii = [c["radius_p99_nm"] for c in report["accepted"]]
clean = locs[~fiducial_mask(locs, centers_nm, radii)]
```

`locs` is the usual `(N, >=4)` array of `[x_nm, y_nm, z_nm, frame, ...]`, and
`field_bounds_nm` is `(x_min, x_max, y_min, y_max)` of the camera. It is
required, because from localizations alone an empty corner and a stray
coordinate look the same; `field_bounds_from_data(locs, max_drift_nm)` returns
a guess when there is no metadata. The `report` carries per-bead diagnostics:
the pair burden, the measured cloud radius (which `fiducial_mask` accepts per
bead), the frame span, and how many bins of pure chance would be expected to
reach that occupancy. Two beads closer together than `max_drift_nm` are
reported as one — that is the resolution limit. See
[docs/fiducials.md](docs/fiducials.md).

## Running the tests

The test suite ships with the package, so a GPU user can validate the CUDA
backend on their own hardware:

```bash
pip install "comet-smlm[test]"
pytest --pyargs comet.tests comet_fiducials.tests
```

Tests needing a GPU are marked and skip automatically.

## Citing COMET

If you use COMET in your research, please cite the
[preprint](https://doi.org/10.64898/2026.03.27.714864). Machine-readable
metadata is in
[CITATION.cff](https://github.com/gpufit/Comet/blob/master/CITATION.cff).

## License

MIT. See [LICENSE.txt](LICENSE.txt).
