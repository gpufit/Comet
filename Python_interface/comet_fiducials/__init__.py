"""Automatic fiducial-bead detection for SMLM localization data.

A standalone package: it imports nothing but ``numpy`` and ``scipy``, and in
particular nothing from COMET.  It ships alongside COMET because that is where
it is used, but it is not part of the drift-correction path -- COMET never
calls it, and it never calls COMET.  ``comet_fiducials/histogram.py`` has no
intra-package imports either, so a single file is enough to vendor it.

See :mod:`comet_fiducials.histogram` for the method.

>>> from comet_fiducials import find_fiducials, fiducial_mask
>>> centers_nm, report = find_fiducials(locs, 250, camera_bounds_nm)  # doctest: +SKIP
>>> clean = locs[~fiducial_mask(locs, centers_nm, 250)]          # doctest: +SKIP
"""

from .histogram import (expected_false_bins, fiducial_mask,
                        field_bounds_from_data, find_fiducials,
                        outer_fence_threshold)

__all__ = [
    "expected_false_bins",
    "field_bounds_from_data",
    "fiducial_mask",
    "find_fiducials",
    "outer_fence_threshold",
]
