# API Reference

A grouped overview of the public functions in pyCOMET. See docstrings for parameter details.

## Drift correction

::: comet.core.drift_optimizer.comet_run_kd
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

### Progress, cancellation and run details

An application that runs COMET for minutes can follow the run, stop it, and
find out afterwards what it did:

```python
from comet import comet_run_kd

class Cancelled(Exception):
    pass

def progress(stage, info):
    if stage == "evaluation":
        print(f"run {info['run']}, sigma {info['sigma_nm']:.1f} nm, "
              f"evaluation {info['n_evaluations']}")
    if user_pressed_cancel():
        raise Cancelled

try:
    drift, details = comet_run_kd(locs.copy(), segmentation_mode=2, segmentation_var=60,
                                  max_drift_nm=300, target_sigma_nm=10,
                                  progress=progress, return_details=True)
except Cancelled:
    ...   # nothing was written to the array passed in
```

The callback is called synchronously from the thread running COMET, at every
stage listed under `progress` above, and after every evaluation of the cost
function — the unit a long run is made of, so a cancel takes effect within one
evaluation. The one step that cannot be interrupted is the neighbour-pair
search. `details` is a `RunDetails`: the windows, the per-window drift (the
knots the per-frame drift is interpolated from), the accepted kernel width, the
numbers of runs, evaluations and pairs, the backend and the time per stage.

::: comet.core.drift_optimizer.RunDetails
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
::: comet.core.drift_optimizer.optimize_3d_chunked_better_moving_avg_kd
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

## Segmentation

::: comet.core.segmenter.segmentation_wrapper
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

## Pair search

::: comet.core.pair_indices.pair_indices_kdtree
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

## Automatic fiducial detection

Standalone: `comet_fiducials` imports nothing from `comet`. See
[Fiducial detection](fiducials.md) for the method and the measurements behind
it.

::: comet_fiducials.find_fiducials
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true
::: comet_fiducials.fiducial_mask
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

## Interpolation

::: comet.core.interpolation.interpolate_drift
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

## I/O utilities

::: comet.core.io_utils.load_thunderstorm_csv
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true
::: comet.core.io_utils.save_dataset_as_ms_h5
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true
::: comet.core.io_utils.save_drift_correction_details
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true
