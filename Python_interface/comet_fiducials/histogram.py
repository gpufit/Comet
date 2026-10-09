"""Fiducial-bead detection from one drift-scale 2D histogram.

The concept, in the order the code runs it
==========================================

**Step 1: one histogram, binned at the drift scale.**  A fiducial is a static
point source, so the only thing that spreads its localizations in uncorrected
data is the stage drift, and the cloud is at most ``max_drift_nm`` across.
Binning at exactly that scale is the unique choice that puts a whole fiducial
in one bin: finer bins shatter it, coarser bins dilute it into the structure
around it.  It also makes the occupancy mean something directly -- ``n``
localizations in a bin are ``n(n-1)/2`` pairable localizations at the pairing
radius.  Only occupied bins are materialised.

**Step 2: the bins chance would not have produced.**  Given the field's own
density there is an occupancy a uniform field would not reach anywhere (see
:func:`expected_false_bins`); every bin above it is a proposal.  Derived from
the data rather than a fixed percentile, so nothing caps how many fiducials can
be found.  The cut is computed twice, the second time from a background
re-estimated outside the first shortlist, so a bright source cannot raise it far
enough to hide a faint one.

**Step 3: merge, seed at local maxima, leave the grid.**  A fiducial on a grid
line lights up two or four bins, so touching proposals are merged
(8-connectivity) and each local-maximum bin inside the blob seeds its own
candidate -- a single window on a blob's joint centre of mass sits between two
nearby fiducials and catches neither.  Each seed is then walked off the grid by
a box mean shift: a bin-sized window recentred on its own centre of mass.

**Step 4: score by cross-time pair mass.**  A fiducial does not merely hold many
localizations, it holds many *spread over the acquisition*, and only pairs whose
members fall in different time blocks carry drift information.  ``pair_mass``
counts those over the candidate's bin-sized square window, treating it as a
complete graph: an upper bound on a radius-limited count -- the square's
diagonal is ``sqrt(2)`` bins and only XY is used -- but it has the closed form
``(n^2 - sum_b n_b^2) / 2``, so it costs one bincount instead of a KD-tree.
``pair_mass_fine`` is the exact radius-limited count when it is wanted.

Every cross-time count is taken as the smaller of **two time grids offset by
half a block**, or a short burst that happens to straddle a boundary scores as
though it were persistent.  Half a block is ``minimum_persistence`` of the
acquisition: a stated policy, and the only timescale in the method.

Selection
---------
A log-space Tukey fence on the pair mass, built from every occupied bin so the
reference always contains ordinary field, floored at ``min_effect_size`` times
the bulk so an outlier has to be large rather than merely unusual.  Plus a
minimum reference size, and the same chance test that produced the shortlist.

Acceptance is graded, because spatial evidence is not the whole story.  A
candidate that passes everything is ``high`` confidence and is returned; one
whose localizations arrive in a few brief episodes far apart -- which no
cross-time count can tell from a persistent source -- is ``needs_review`` and is
reported but not returned.

Limits
------
Resolution is one to two bins: fiducials closer than ``max_drift_nm`` come back
as one centre, two bins apart they always resolve, in between it depends on the
grid phase.  A short bright burst is demoted two orders of magnitude by the pair
mass but can still clear a relative fence in a quiet field.

``docs/fiducials.md`` carries the measurements behind every choice here, and
the head-to-head against Picasso's fiducial finder.

Examples
--------
>>> from comet_fiducials import fiducial_mask, find_fiducials
>>> centers, report = find_fiducials(locs, 250, camera_bounds_nm)  # doctest: +SKIP
>>> radii = [c["radius_p99_nm"] for c in report["accepted"]]       # doctest: +SKIP
>>> clean = locs[~fiducial_mask(locs, centers, radii)]             # doctest: +SKIP
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import poisson

__all__ = [
    "expected_false_bins",
    "field_bounds_from_data",
    "fiducial_mask",
    "find_fiducials",
    "outer_fence_threshold",
]

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Every constant the method carries, in one place, with what it is for.  The
# measurements behind each value are in docs/fiducials.md.
# ---------------------------------------------------------------------------
# Tukey's outer fence: 1.5 is the inner fence, 3 the outer and conservative one.
_FENCE_IQR_FACTOR = 3.0
# Minimum effect size, applied to every reference rather than to a special
# case.  Pair mass scales as occupancy squared, so 25 means "a fiducial bin
# holds five times the localizations of ordinary field".  A floor under the
# fence, never a cap on how many fiducials may be named.  It is the dial
# between false positives and faint-fiducial sensitivity, and 25 is the knee of
# the measured curve: the smallest floor that clears the false positives.
_MIN_EFFECT_SIZE = 25.0
# Box mean shift: stop when the centre moves less than this fraction of a bin,
# or after this many passes.  Relative, so it scales with max_drift_nm.
_RECENTER_TOL_FRACTION = 0.004
_RECENTER_MAX_PASSES = 8
# Radius for the optional KD-tree diagnostics only.  Drift scale is a poor
# proxy for localization precision; nothing in the default path reads it.
_FINE_RADIUS_FRACTION = 0.125
# Derived field bounds keep this central fraction of the localizations per
# axis, grown by one bin so a source at the very edge is not clipped.
_FIELD_QUANTILE = 0.999
# The source is measured inside a disc of this many bin radii and the local
# background in the ring just outside it.  Two bins for the source because that
# is the resolution limit; one bin of ring beyond, with five times the area.  A
# one-bin disc truncates a cloud wider than max_drift_nm while its tail sits in
# the ring and inflates the background, so removal silently leaves part of it.
_SOURCE_DISC = 2.0
_BACKGROUND_ANNULUS = 3.0
# Angular sectors of that ring.  The density is their median, so a single
# contaminating source occupies one and does not move the estimate.
_BACKGROUND_SECTORS = 8
# Polar sample grid used to measure how much of a disc or sector is really
# inside the field, rather than assuming all of it is.
_AREA_ANGLES = 64
_AREA_STEPS = 128
# A sector this much smaller than the largest is too clipped by the field edge
# to give a usable density, and is left out of the median.
# A bin at least this much inside the field is ordinary field and belongs in
# the reference; a sliver holds a fraction of a bin's localizations and the
# square of that fraction of its pairs, and mixing the two made the fence
# bimodal -- on a field a fractional number of bins across, the log-space IQR
# spanned both modes and the fence rose into the billions.
_MIN_REFERENCE_EXPOSURE = 0.5
# Shortlisted bins in a seed's own 3x3 neighbourhood that a source no wider
# than one bin can account for, at any grid phase: a 2x2 block.  Measured
# there rather than over the merged component, which does not exist when
# merge_edges is off and made every candidate look compact.
_COMPACT_BINS = 4
_MIN_OBSERVABLE_SECTOR = 0.25
# Floor on the blocks temporal_entropy uses.  A floor, not a fixed count:
# entropy must not inherit a coarse scoring grid (at two blocks everything
# looks uniform), nor be coarser than it, or a source shorter than one entropy
# block is refused however short minimum_persistence says a persistent source
# may be.
_ENTROPY_BLOCKS_MIN = 64
# Percentile of the background-subtracted cloud reported as the removal radius.
_CLOUD_PERCENTILE = 99.0
# Below this fraction of the temporal spread a contiguous source of the same
# span would show, a candidate needs review rather than automatic removal.  A
# continuously present source scores ~1 however briefly it was present; two
# brief episodes far apart score ~0.2.
_MIN_TEMPORAL_COVERAGE = 0.5
# A field needing more bins than this to index is a coordinate error, not a
# field of view.  Only occupied bins are materialised, so the limit exists to
# keep the bin key inside int64.
_MAX_GRID_CELLS = 2 ** 62

# --------------------------------------------------------------------------
# selection thresholds
# --------------------------------------------------------------------------
def outer_fence_threshold(scores, iqr_factor=_FENCE_IQR_FACTOR,
                          min_effect_size=_MIN_EFFECT_SIZE):
    """High-outlier threshold on a positive, heavy-tailed score.

    Tukey's outer fence in log space, ``expm1(q3 + iqr_factor * IQR)`` of
    ``log1p(scores)``, floored at ``min_effect_size`` times the bulk.  The floor
    is what makes it safe: on a sparse field, where bin pair masses take only a
    handful of distinct values, the fence alone sits a few percent above the
    bulk and names a whole dense region.  ``inf`` -- name nothing -- only when
    there are fewer than two positive scores to place a fence against.

    Parameters
    ----------
    scores : array_like
        Positive scores, one per bin.  Non-positive entries are ignored.
    iqr_factor : float, optional
        Multiple of the interquartile range added to the upper quartile.
    min_effect_size : float, optional
        How far above the bulk an outlier must sit.  Pair mass scales as
        occupancy squared, so the default means five times the localizations of
        ordinary field.

    Returns
    -------
    float
        The threshold, or ``inf``.

    Notes
    -----
    Whether anything clears the fence is the caller's decision, made against
    the scores it will actually judge.  Deciding it here would mean deciding it
    against the reference, and the reference is not in the same unit: a
    candidate gathers a bin *and its halo* and is recentred on the source, so a
    fiducial split across a grid crossing carries a mass that no one of its own
    raw bins does.  Returning ``inf`` because no single bin cleared the fence
    threw exactly that fiducial away.

    The bulk is an upper quartile, so this carries a prevalence assumption:
    once sources exceed a quarter of the positive reference, ``q3`` sits inside
    them and the threshold rises above every source.  Contamination pushes it
    *up*, so the failure is a fence that is too strict.
    :func:`find_fiducials` reports ``reference_shortlist_fraction`` and names
    the case in ``fence_undefined_reason``.

    Examples
    --------
    >>> float(outer_fence_threshold([1, 1, 2, 2, 3, 1000]))
    68.75
    >>> float(outer_fence_threshold([5, 5, 5, 5]))                # no spread
    125.0
    >>> float(outer_fence_threshold([10.0] * 1545 + [105.0] * 55))  # a second mode
    250.0
    >>> float(outer_fence_threshold([10.0] * 400 + [1e6] * 12))   # a real tail
    250.0
    """
    if not (np.isfinite(min_effect_size) and float(min_effect_size) >= 1.0):
        raise ValueError(
            "min_effect_size must be finite and at least 1 (a floor below the "
            "bulk is not a floor); got %r." % (min_effect_size,))
    scores = np.asarray(scores, dtype=float)
    positive = scores[scores > 0]
    if positive.size < 2:
        return np.inf

    log_scores = np.log1p(positive)
    q1, q3 = np.quantile(log_scores, (0.25, 0.75))
    bulk = float(np.quantile(positive, 0.75))
    tukey = float(np.expm1(q3 + iqr_factor * (q3 - q1)))
    return max(tukey, float(min_effect_size) * max(bulk, 1.0))


def expected_false_bins(peak_bin_locs, background_rate, n_bins):
    """How many bins this occupancy would be expected to reach by chance.

    ``n_bins * P(Poisson(lambda) >= n)``: the expected count of bins in the
    whole field that chance alone would push this high.  Below 1 means the
    occupancy is not something the field would have thrown up on its own.  A
    log-space fence has no notion of how many bins it looked at, and the
    maximum of tens of thousands of Poisson draws sails past one.

    Parameters
    ----------
    peak_bin_locs : int
        Occupancy of the candidate's fullest **grid** bin.  A grid count, not
        the recentred window's: the window is placed to maximise its own
        content, so it is a maximum over a continuum of placements and
        ``n_bins`` would understate the trials.
    background_rate : float
        Expected localizations in **this** bin: the rate over all bins of the
        field, including empty ones -- the same population ``n_bins`` counts --
        times the bin's own share of a whole one.  An occupied-bin average
        would be zero-truncated and far too high on a sparse field.  That share
        is 1 everywhere but the boundary of the field, so
        :func:`find_fiducials` passes ``background_rate_locs_per_bin`` times the
        candidate's ``peak_bin_exposure``; reproducing a reported value without
        that factor overstates lambda for a boundary bin.
    n_bins : int
        Number of bins the maximum was taken over.

    Returns
    -------
    float
        Expected number of chance bins at or above this occupancy.

    Examples
    --------
    >>> round(expected_false_bins(30, 2.0, 40_000), 4)   # a Poisson maximum
    0.0
    >>> expected_false_bins(9, 2.0, 40_000) > 1          # ordinary fluctuation
    True
    >>> expected_false_bins(5, 0.05, 150_000) < 0.01     # sparse field, real source
    True
    """
    if background_rate <= 0 or peak_bin_locs <= 0:
        return 0.0
    log_expected = np.log(max(n_bins, 1)) + poisson.logsf(
        int(peak_bin_locs) - 1, background_rate)
    return float(np.exp(min(log_expected, 700.0)))


def _as_whole_number(value, name, minimum, allow_none=False):
    """Validate an integer-valued parameter without silently truncating it."""
    if value is None:
        if allow_none:
            return None
        raise ValueError("%s must be an integer of at least %d." % (name, minimum))
    if not np.isfinite(value) or float(value) != int(value):
        raise ValueError("%s must be a finite whole number; got %r." % (name, value))
    if int(value) < minimum:
        raise ValueError("%s must be at least %d; got %d." % (name, minimum, int(value)))
    return int(value)


def _chance_occupancy(background_rate, n_bins, level):
    """Smallest bin occupancy that chance would not be expected to reach."""
    if background_rate <= 0 or n_bins <= 0:
        return 2
    threshold = poisson.isf(min(level / n_bins, 1.0), background_rate)
    return max(int(threshold) + 1, 2)


def _occupancy_by_exposure(background_rate, exposure, n_bins, level):
    """The same cut per cell, each at its own share of a whole bin.

    A rectangular field gives exposure a handful of distinct values -- one for
    the interior and one for each clipped edge -- so the Poisson inversion is
    done once per value rather than once per cell.
    """
    cuts = np.empty(len(exposure), dtype=np.int64)
    for share in np.unique(exposure):
        cuts[exposure == share] = _chance_occupancy(
            background_rate * float(share), n_bins, level)
    return cuts


# --------------------------------------------------------------------------
# cross-time pair counting
# --------------------------------------------------------------------------
def _time_block_grids(frames, block_width):
    """Two block labellings of the given width, offset by half a block."""
    frames = np.asarray(frames, dtype=np.int64)
    shifted = frames - int(frames.min())
    width = max(1, int(block_width))
    return (np.floor_divide(shifted, width),
            np.floor_divide(shifted + width // 2, width))


def _sum_of_squared_block_counts(block_labels):
    """Sum of squared per-block counts, sparse in the block axis.

    ``bincount`` would allocate the whole label range, which a large frame gap
    makes enormous.
    """
    _, counts = np.unique(block_labels, return_counts=True)
    counts = counts.astype(np.float64)
    return float(np.dot(counts, counts)), float(counts.sum())


def _analytic_cross_time_pairs(blocks_a, blocks_b):
    """Complete-graph cross-time pair count, conservative over both time grids.

    ``(n^2 - sum_b n_b^2) / 2`` on each grid; the smaller is returned.  See the
    module docstring for what this counts and what it does not.
    """
    if len(blocks_a) < 2:
        return 0.0
    square_a, total = _sum_of_squared_block_counts(blocks_a)
    square_b, _ = _sum_of_squared_block_counts(blocks_b)
    return float((total * total - max(square_a, square_b)) / 2.0)


def _time_uniform_subsample(frames, max_locs):
    """Indices of an every-k-th-in-time subsample, plus the stride k.

    Striding in time preserves the cross-time *fraction*, so a count on the
    subsample times ``k ** 2`` estimates the full one.  Deterministic, hence
    reproducible in a QC report, but it can alias frame-periodic data.
    """
    n = len(frames)
    if max_locs is None or n <= int(max_locs):
        return np.arange(n), 1
    stride = int(np.ceil(n / float(max_locs)))
    order = np.argsort(frames, kind="stable")
    return order[::stride], stride


def _kd_cross_time_pairs(xy, blocks_a, blocks_b, radius_nm):
    """Cross-time pairs within ``radius_nm``, counted without listing them.

    ``count_neighbors`` returns ordered pairs including ``i == j``, hence the
    ``(count - n) / 2`` conversion.  The whole-cloud count is grid-independent,
    so only the same-block part is recomputed per grid.
    """
    if len(xy) < 2:
        return 0.0
    tree = cKDTree(xy)
    total = (tree.count_neighbors(tree, radius_nm) - len(xy)) / 2.0
    same = []
    for block_labels in (blocks_a, blocks_b):
        subtotal = 0.0
        for block in np.unique(block_labels):
            member = xy[block_labels == block]
            if len(member) < 2:
                continue
            block_tree = cKDTree(member)
            subtotal += (block_tree.count_neighbors(block_tree, radius_nm)
                         - len(member)) / 2.0
        same.append(subtotal)
    return float(total - max(same))


def _cell_exposure(cell_x, cell_y, origin, bin_nm, field_bounds):
    """Share of a whole bin each cell has inside the field, in [0, 1]."""
    x_min, x_max, y_min, y_max = field_bounds
    lower = np.asarray(origin, dtype=float) + np.column_stack(
        [cell_x, cell_y]).astype(float) * bin_nm
    return np.prod(np.clip((np.minimum(lower + bin_nm, [x_max, y_max])
                            - np.maximum(lower, [x_min, y_min])) / bin_nm,
                           0.0, 1.0), axis=1)


def _all_bin_pair_mass(cell_index, blocks_a, blocks_b, n_occupied):
    """Cross-time pair mass of every occupied bin, over both time grids.

    Sparse in both axes: the temporaries scale with the *observed*
    (bin, block) pairs, never with ``n_occupied * n_blocks``.
    """
    totals = np.bincount(cell_index, minlength=n_occupied).astype(np.float64)
    squares = None
    for block_labels in (blocks_a, blocks_b):
        n_blocks = int(block_labels.max()) + 1
        combined = cell_index.astype(np.int64) * n_blocks + block_labels
        observed, counts = np.unique(combined, return_counts=True)
        per_grid = np.bincount(
            observed // n_blocks,
            weights=counts.astype(np.float64) ** 2,
            minlength=n_occupied,
        )
        squares = per_grid if squares is None else np.maximum(squares, per_grid)
    return (totals * totals - squares) / 2.0


# --------------------------------------------------------------------------
# field of view
# --------------------------------------------------------------------------
def field_bounds_from_data(dataset, max_drift_nm, quantile=_FIELD_QUANTILE):
    """Guess a field of view from the localizations alone.

    :func:`find_fiducials` requires ``field_bounds_nm`` rather than guessing,
    because from localizations alone an empty corner of the camera and an
    invalid coordinate look identical -- and the two call for opposite
    treatment.  Guessing wrong is not benign either way: too large and one
    stray coordinate drives the background rate towards zero, so every
    ordinary fluctuation starts to look significant; too small and a genuine
    fiducial sitting in an empty part of the field is silently discarded.

    Use the camera field of view when you have it.  This helper exists so that
    falling back to a guess is a visible, deliberate line of code.

    Parameters
    ----------
    dataset : (N, >=2) array_like
        Localizations; only the first two columns are read.
    max_drift_nm : float
        Bin size, used to round the box out to whole bins.
    quantile : float, optional
        Central fraction of the localizations to keep per axis.

    Returns
    -------
    tuple of float
        ``(x_min, x_max, y_min, y_max)``, rounded out to whole bins and grown
        by one bin so a source at the edge of the data is not clipped.

    Examples
    --------
    >>> bounds = field_bounds_from_data(locs, 250)             # doctest: +SKIP
    >>> centers, report = find_fiducials(locs, 250, bounds)    # doctest: +SKIP
    """
    dataset = np.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[1] < 2:
        raise ValueError("dataset must have shape (N, >=2).")
    bin_nm = float(max_drift_nm)
    if not np.isfinite(bin_nm) or bin_nm <= 0:
        raise ValueError("max_drift_nm must be a positive, finite number.")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile must be in (0, 1].")
    # Only x and y are read, so only those are converted: the columns
    # documented as ignored are neither copied nor required to be numeric.
    xy = np.column_stack([dataset[:, 0].astype(np.float64),
                          dataset[:, 1].astype(np.float64)])
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        raise ValueError("dataset contains no finite coordinates.")
    tail = (1.0 - float(quantile)) / 2.0
    low = np.floor(np.quantile(xy, tail, axis=0) / bin_nm) * bin_nm - bin_nm
    high = np.ceil(np.quantile(xy, 1.0 - tail, axis=0) / bin_nm) * bin_nm + bin_nm
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def _resolve_field_bounds(field_bounds_nm):
    """Validate the field of view.  There is deliberately no default."""
    if field_bounds_nm is None:
        raise ValueError(
            "field_bounds_nm is required: the Poisson trial count and the "
            "background rate both describe a field of view, and it cannot be "
            "inferred safely from localizations alone -- an empty corner of "
            "the camera and an invalid coordinate look the same. Pass the "
            "camera field of view, or field_bounds_from_data(dataset, "
            "max_drift_nm) to accept a guess explicitly.")
    bounds = np.asarray(field_bounds_nm, dtype=float).ravel()
    if bounds.size != 4 or not np.all(np.isfinite(bounds)):
        raise ValueError(
            "field_bounds_nm must be four finite numbers "
            "(x_min, x_max, y_min, y_max).")
    x_min, x_max, y_min, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("field_bounds_nm must have x_max > x_min and y_max > y_min.")
    return float(x_min), float(x_max), float(y_min), float(y_max)




# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def find_fiducials(
    dataset,
    max_drift_nm,
    field_bounds_nm,
    minimum_persistence=1.0 / 64.0,
    max_expected_false_bins=1.0,
    min_effect_size=_MIN_EFFECT_SIZE,
    min_reference_bins=16,
    max_locs_per_candidate=20_000,
    compute_fine_pair_mass=False,
    grid_offset_nm=(0.0, 0.0),
    merge_edges=True,
    return_report=True,
):
    """Locate fiducials from one drift-scale 2D histogram.

    ``dataset``, ``max_drift_nm`` and ``field_bounds_nm`` are required; every
    other argument has a derived default.  See the module docstring for the
    method and its limits.

    Parameters
    ----------
    dataset : (N, >=4) array_like
        Localizations as ``[x_nm, y_nm, z_nm, frame, ...]``.  Only x, y and
        frame are read; z and further columns are ignored and need not be
        numeric.  Rows with a non-finite x, y or frame are dropped.
    max_drift_nm : float
        Maximum expected drift -- the number COMET takes as its pair radius.
        Doubles as the histogram bin size and as the window the pair mass is
        counted over, and it is the **resolution limit**: two fiducials closer
        together than this are reported as one.
    field_bounds_nm : (4,) array_like
        ``(x_min, x_max, y_min, y_max)`` of the field of view.  Localizations
        outside it are discarded before anything else runs, so every stage
        describes the same field the Poisson trial count does.  There is no
        default -- it cannot be inferred safely, since from localizations alone
        an empty corner of the camera and an invalid coordinate look identical.
        :func:`field_bounds_from_data` returns a guess when there is no
        metadata, so accepting one is an explicit call.
    minimum_persistence : float in (0, 0.5), exclusive, optional
        Fraction of the acquisition a source must span, contiguously, before
        its localizations count as spread over time.  A stated policy, and the
        only timescale in the method; the block width comes straight from it
        and the value achieved after rounding to whole frames is reported as
        ``effective_minimum_persistence``.  0.5 is excluded: it leaves one
        block, one time grid with no cross-block pairs, and every pair mass
        zero.
    max_expected_false_bins : float, optional
        Selection level, in expected number of bins in the whole field that
        chance alone would push this high (see :func:`expected_false_bins`).
        Sets both the proposal shortlist and the per-candidate chance test;
        ``None`` disables both, so every bin that can carry a cross-time pair
        is proposed and the fence alone decides.
    min_effect_size : float, optional
        How far above the bulk of the field a candidate's pair mass must sit to
        be an outlier at all -- the floor under the fence, and the dial between
        false positives and faint-fiducial sensitivity.
    min_reference_bins : int, optional
        Refuse to name anything unless the fence reference holds at least this
        many bins with a positive pair mass.  An outlier test needs a sample:
        with one or two bins, *everything* looks like an outlier.  0 opts out.
    max_locs_per_candidate : int or None, optional
        Above this, a candidate is subsampled uniformly in time and its
        KD-tree count rescaled by ``stride ** 2``.  ``None`` disables
        subsampling, which is affordable only on small candidates.
    compute_fine_pair_mass : bool, optional
        Compute the KD-tree diagnostics ``pair_mass_fine`` and
        ``concentration``.  Off by default -- nothing in the selection reads
        them and they are the bulk of the runtime.  ``nan`` when skipped.
    grid_offset_nm : pair of float, optional
        Shifts the histogram origin.  Exists so the grid phase can be swept in
        a test; normalised modulo the bin, so whole-bin offsets are identical.
    merge_edges : bool, optional
        Merge touching proposals before centring.  ``False`` is the naive
        one-candidate-per-bin behaviour, kept for the ablation.
    return_report : bool, optional
        When ``False``, return only the centres.

    Returns
    -------
    centers_nm : (M, 2) ndarray
        High-confidence fiducial centres in nm, strongest first.  Candidates
        held for review are in ``report["needs_review"]``, not here.
    report : dict
        Returned unless ``return_report=False``.

        ``accepted``, ``needs_review``, ``rejected``
            Per-candidate dicts (below), each list sorted by pair mass.  Only
            ``accepted`` appears in ``centers_nm``.
        ``score_threshold``, ``min_effect_size``, ``n_reference_bins``,
        ``min_reference_bins``, ``insufficient_reference``,
        ``reference_shortlist_fraction``, ``fence_undefined_reason``
            Where the fence landed, the floor under it, its sample size, and
            why nothing was named if nothing was: ``high_shortlist_fraction``
            when contamination pushed the fence above its own sources, and
            ``no_separation`` when the fence simply stood.
        ``field_bounds_nm``, ``n_bins``, ``n_locs_in_field``,
        ``n_locs_outside_field``, ``background_rate_locs_per_bin``,
        ``chance_occupancy``, ``first_pass_occupancy``,
        ``max_expected_false_bins``
            The field of view, what it cost, the density in it, and the derived
            proposal cut before and after the background was re-estimated.
        ``bin_size_nm``, ``n_occupied_bins``, ``n_selected_bins``,
        ``n_components``
            The histogram and the sizes at each stage.
        ``time_block_frames``, ``n_time_blocks``, ``minimum_persistence``,
        ``effective_minimum_persistence``, ``entropy_block_frames``,
        ``n_entropy_blocks``, ``pair_radius_nm``, ``compute_fine_pair_mass``,
        ``grid_offset_nm``, ``merge_edges``
            The remaining settings, echoed.

        Each candidate dict holds ``center_nm``, ``n_locs``, ``footprint_bins``
        (shortlisted bins in its own 3x3 neighbourhood -- how much of the grid
        the source occupies, and unlike ``n_bins`` independent of
        ``merge_edges``), ``peak_bin_locs``,
        ``peak_bin_exposure`` (that bin's share of a whole one, below 1 only on
        the boundary of the field),
        ``pair_mass``, ``pair_mass_strided``, ``pair_mass_fine``,
        ``concentration``, ``subsample_stride``, ``n_bins``, ``grid_shift_nm``,
        ``radius_p99_nm``, ``n_background_sectors``, ``window_frame_span``,
        ``source_frame_span``,
        ``n_time_blocks_present``, ``temporal_entropy``,
        ``temporal_coverage``, ``confidence`` and ``expected_false_bins``;
        anything not accepted also carries ``reject_reason``.

        ``radius_p99_nm`` is the cloud radius with the local background
        subtracted and can be passed straight to :func:`fiducial_mask`.
        ``window_frame_span`` is the raw extent of everything in the window;
        ``source_frame_span`` is where the source's own mass is, and is what a
        reviewer should be shown.  ``temporal_coverage`` is the source's
        background-subtracted temporal entropy over the entropy a *contiguous*
        source of the same span would show -- ~1 for continuous presence
        however brief, ~0.2 for disconnected episodes -- and it is that ratio,
        not the raw entropy, which decides ``high`` against ``needs_review``.

    Raises
    ------
    ValueError
        If ``dataset`` has fewer than four columns, no finite rows, or nothing
        inside the field; if any parameter is out of range; if
        ``field_bounds_nm`` or ``grid_offset_nm`` is malformed; or if the
        coordinate extent divided by ``max_drift_nm`` is too large to index.

    See Also
    --------
    fiducial_mask : Turn the returned centres into a removal mask.
    field_bounds_from_data : Guess a field of view when there is no metadata.

    Examples
    --------
    >>> centers, report = find_fiducials(locs, 250.0, camera_bounds_nm)  # doctest: +SKIP
    >>> [c["radius_p99_nm"] for c in report["accepted"]]                 # doctest: +SKIP
    [30.2, 28.9]
    """
    dataset = np.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[1] < 4:
        raise ValueError("dataset must have shape (N, >=4): [x_nm, y_nm, z_nm, frame, ...].")
    # Only x, y and frame are read, so only those are converted: the columns
    # documented as ignored are neither copied nor required to be numeric.
    dataset = np.column_stack([
        dataset[:, 0].astype(np.float64), dataset[:, 1].astype(np.float64),
        dataset[:, 3].astype(np.float64)])
    bin_nm = float(max_drift_nm)
    if not np.isfinite(bin_nm) or bin_nm <= 0:
        raise ValueError("max_drift_nm must be a positive, finite number.")
    if not 0 < float(minimum_persistence) < 0.5:
        # At 0.5 the whole acquisition is one block, one of the two time grids
        # has no cross-block pairs at all, and every pair mass collapses to zero.
        raise ValueError("minimum_persistence must be in (0, 0.5), exclusive.")
    if max_expected_false_bins is not None and not (
            np.isfinite(max_expected_false_bins) and float(max_expected_false_bins) > 0):
        raise ValueError(
            "max_expected_false_bins must be positive and finite, or None to disable.")
    max_locs_per_candidate = _as_whole_number(
        max_locs_per_candidate, "max_locs_per_candidate", minimum=2, allow_none=True)
    min_reference_bins = _as_whole_number(
        min_reference_bins, "min_reference_bins", minimum=0)
    if not (np.isfinite(min_effect_size) and float(min_effect_size) >= 1.0):
        raise ValueError(
            "min_effect_size must be finite and at least 1 (a floor below the "
            "bulk is not a floor); got %r." % (min_effect_size,))
    pair_radius_nm = bin_nm * _FINE_RADIUS_FRACTION

    offset = np.asarray(grid_offset_nm, dtype=float).ravel()
    if offset.size != 2 or not np.all(np.isfinite(offset)):
        raise ValueError("grid_offset_nm must be two finite numbers.")
    # Modulo the bin: whole-bin shifts are the same grid, and a negative offset
    # must not be able to push cell coordinates below the origin.
    offset = np.mod(offset, bin_nm)

    # z is documented as ignored, so a missing z must not cost the row.
    finite = np.isfinite(dataset).all(axis=1)
    dataset = dataset[finite]
    if len(dataset) == 0:
        raise ValueError("dataset contains no finite localizations.")
    # ---- the statistical field: everything below is judged inside it -------
    x_min, x_max, y_min, y_max = _resolve_field_bounds(field_bounds_nm)
    # Half-open in both axes, so a localization exactly on the upper bound
    # belongs to no cell and the bin count is exactly the field's.
    in_field = ((dataset[:, 0] >= x_min) & (dataset[:, 0] < x_max)
                & (dataset[:, 1] >= y_min) & (dataset[:, 1] < y_max))
    n_locs_outside_field = int((~in_field).sum())
    dataset = dataset[in_field]
    if len(dataset) == 0:
        raise ValueError("no localizations fall inside field_bounds_nm.")
    xy = dataset[:, :2]
    frames = np.rint(dataset[:, 2]).astype(np.int64)
    # Block width straight from the requested fraction, so the effective
    # persistence is not silently rounded to whatever an integer block count
    # happens to give.  It is reported back.
    frame_span = int(frames.max()) - int(frames.min()) + 1
    block_width = max(1, int(round(2.0 * float(minimum_persistence) * frame_span)))
    blocks_a, blocks_b = _time_block_grids(frames, block_width)
    n_time_blocks = int(np.ceil(frame_span / float(block_width)))
    effective_persistence = block_width / (2.0 * frame_span)
    entropy_blocks_target = max(_ENTROPY_BLOCKS_MIN, n_time_blocks)
    entropy_width = max(1, int(np.ceil(frame_span / float(entropy_blocks_target))))
    n_entropy_blocks = int(np.ceil(frame_span / float(entropy_width)))

    # ---- step 1: one 2D histogram at the drift scale -----------------------
    # Anchored to the field, not to the smallest observed coordinate, so the
    # grid the candidates come from is the same grid the Poisson trial count
    # describes.  The offset shifts the anchor, which can add one cell per
    # axis; that cell is part of the field's grid and is counted as such.
    origin = np.array([x_min, y_min], dtype=float) - offset
    cells = np.floor((xy - origin) / bin_nm).astype(np.int64)
    n_x = int(np.ceil((x_max - origin[0]) / bin_nm))
    n_y = int(np.ceil((y_max - origin[1]) / bin_nm))
    n_x = max(n_x, int(cells[:, 0].max()) + 1)
    n_y = max(n_y, int(cells[:, 1].max()) + 1)
    if n_x * n_y > _MAX_GRID_CELLS:
        raise ValueError(
            "coordinate extent %.3g x %.3g nm at bin %.3g nm needs %d x %d bins; "
            "this is a coordinate error rather than a field of view."
            % ((n_x * bin_nm), (n_y * bin_nm), bin_nm, n_x, n_y))
    linear = cells[:, 0] * n_y + cells[:, 1]
    cell_ids, cell_index, counts = np.unique(
        linear, return_inverse=True, return_counts=True)
    cell_index = cell_index.ravel()
    n_occupied = len(cell_ids)
    lookup = {int(cell_id): position for position, cell_id in enumerate(cell_ids)}

    # One sort gives O(1) access to the members of any occupied cell later on.
    order = np.argsort(cell_index, kind="stable")
    cell_start = np.concatenate(([0], np.cumsum(counts)))

    # ---- step 2: the bins chance would not have produced -------------------
    # The Poisson trial count is a property of the field, not of where the
    # histogram lines fall: n_x and n_y index a grid that can spill one cell
    # past the field when grid_offset_nm is fractional, and counting those
    # would move lambda and the chance cut with the grid phase.
    field_exposure = ((x_max - x_min) / bin_nm) * ((y_max - y_min) / bin_nm)
    n_total_bins = max(1, int(round(field_exposure)))
    n_locs_in_field = int(len(dataset))

    # A cell on the boundary holds only the part of a bin that lies inside the
    # field and cannot be tested as though it held a whole one: on a field a
    # fractional number of bins across every edge cell is a sliver, and calling
    # each one a full trial rejected sources in the middle of it.  Exposure is
    # that share.  Rates stay per whole bin; each cell is cut at its own share.
    cell_x, cell_y = np.divmod(cell_ids, n_y)
    exposure = _cell_exposure(cell_x, cell_y, origin, bin_nm,
                              (x_min, x_max, y_min, y_max))

    # Two passes.  The first rate is inflated by the sources themselves, so a
    # bright one can raise the cut enough to hide a faint one; the second is
    # estimated outside the first shortlist.  Two, not more -- iterating to
    # convergence diverges on structured, non-Poisson fields.
    background_rate = n_locs_in_field / field_exposure
    if max_expected_false_bins is None:
        # Disabled means disabled: every bin that can carry a cross-time pair
        # is proposed, and the fence alone decides.  Leaving the shortlist cut
        # in place would have made "None" disable only half of the test.
        first_pass_occupancy = chance_occupancy = 2
        cell_occupancy = np.full(n_occupied, 2, dtype=np.int64)
        bright = np.empty(0, dtype=np.int64)
        level = np.inf
    else:
        level = float(max_expected_false_bins)
        first_pass_occupancy = _chance_occupancy(background_rate, n_total_bins, level)
        bright = np.flatnonzero(counts >= _occupancy_by_exposure(
            background_rate, exposure, n_total_bins, level))
        background_locs = float(counts.sum() - counts[bright].sum())
        # Exposure on both sides of the division: subtracting a *count* of
        # bright cells from an area measured in bins mixed the two units.
        background_rate = background_locs / max(
            field_exposure - float(exposure[bright].sum()), 1.0)
        chance_occupancy = _chance_occupancy(background_rate, n_total_bins, level)
        cell_occupancy = _occupancy_by_exposure(
            background_rate, exposure, n_total_bins, level)
    selected = np.flatnonzero(counts >= cell_occupancy)
    shortlisted_bin = np.zeros(n_occupied, dtype=bool)
    shortlisted_bin[selected] = True

    # ---- step 3: merge, seed at local maxima, leave the grid ---------------
    components = (
        _connected_components(selected, cell_ids, lookup, n_y) if merge_edges
        else [[int(position)] for position in selected]
    )
    recenter_tol_nm = _RECENTER_TOL_FRACTION * bin_nm

    # Every seed first, so each one can be told where the others are: a
    # neighbouring candidate inside the measuring disc is another source, and
    # must not be counted as this one's background or as its cloud.
    seeds = [(component, seed) for component in components
             for seed in _component_seeds(component, counts, cell_ids, lookup, n_y)]
    seed_centres = np.array(
        [[(cell_ids[seed] // n_y + 0.5) * bin_nm + origin[0],
          (cell_ids[seed] % n_y + 0.5) * bin_nm + origin[1]] for _, seed in seeds]
    ) if seeds else np.empty((0, 2))
    seed_tree = cKDTree(seed_centres) if len(seed_centres) else None

    candidates = []
    for seed_index, (component, seed) in enumerate(seeds):
        # Only seeds that will survive fusion as *separate* candidates count as
        # neighbours.  Seeds within one bin are the same source split across
        # bins and are fused below, so masking them out would carve a hole in
        # this source's own cloud.
        here = seed_centres[seed_index]
        neighbours = (seed_tree.query_ball_point(
            here, r=_BACKGROUND_ANNULUS * bin_nm + bin_nm)
            if seed_tree is not None else [])
        other_centres = np.array(
            [seed_centres[j] for j in neighbours
             if j != seed_index
             and np.linalg.norm(seed_centres[j] - here) > bin_nm]
        ) if neighbours else np.empty((0, 2))
        # How much of the grid this source actually occupies, read off its own
        # neighbourhood: a source no wider than a bin covers a 2x2 block at any
        # phase, ordinary dense tissue fills all nine.  Not the merged
        # component -- with merge_edges off there is none, and every candidate
        # then claimed to be compact.
        around = _neighbour_positions(seed, cell_ids, lookup, n_y, radius=1)
        footprint_bins = (1 + int(np.count_nonzero(shortlisted_bin[around]))
                          if around else 1)
        halo = _seed_halo(seed, cell_ids, lookup, n_y, order, cell_start)
        halo_xy = xy[halo]
        member_idx = order[cell_start[seed]:cell_start[seed + 1]]
        centre = xy[member_idx].mean(axis=0)
        for _ in range(_RECENTER_MAX_PASSES):
            inside = np.all(np.abs(halo_xy - centre) <= bin_nm / 2.0, axis=1)
            if not np.any(inside):
                break
            new_centre = halo_xy[inside].mean(axis=0)
            shifted = float(np.linalg.norm(new_centre - centre))
            centre = new_centre
            if shifted <= recenter_tol_nm:
                break
        inside = np.all(np.abs(halo_xy - centre) <= bin_nm / 2.0, axis=1)
        window_idx = halo[inside] if np.any(inside) else member_idx
        # The radius is measured over a disc of _SOURCE_DISC bins plus the
        # ring beyond it that the background is estimated from.
        near = np.linalg.norm(halo_xy - centre, axis=1) <= _BACKGROUND_ANNULUS * bin_nm
        cloud_idx = halo[near] if np.any(near) else window_idx

        candidates.append(_score_candidate(
            centre=centre, window_idx=window_idx, cloud_idx=cloud_idx,
            xy=xy, frames=frames, blocks_a=blocks_a, blocks_b=blocks_b,
            bin_nm=bin_nm, pair_radius_nm=pair_radius_nm,
            max_locs=max_locs_per_candidate,
            compute_fine_pair_mass=bool(compute_fine_pair_mass),
            n_bins=len(component), footprint_bins=footprint_bins,
            peak_bin_locs=int(counts[seed]),
            peak_bin_exposure=float(exposure[seed]),
            entropy_width=entropy_width, n_entropy_blocks=n_entropy_blocks,
            frame_min=int(frames.min()), frame_max=int(frames.max()),
            background_per_block=background_rate / max(n_entropy_blocks, 1),
            field_bounds=(x_min, x_max, y_min, y_max),
            other_centres=other_centres,
            grid_shift_nm=float(np.linalg.norm(centre - xy[member_idx].mean(axis=0)))))

    # A fiducial whose cloud straddles a grid line can surface as two seeds
    # whose recentred windows then coincide.  Fuse anything closer than one
    # bin, keeping the stronger.  This is also the resolution limit.
    candidates = _fuse_coincident(candidates, bin_nm)

    # ---- step 4: fence the pair mass against the whole field -------------
    # Only whole bins: the fence describes what ordinary field looks like, and
    # a bin the field boundary cuts in half is a partial observation of it.
    whole = exposure >= _MIN_REFERENCE_EXPOSURE
    reference = _all_bin_pair_mass(
        cell_index, blocks_a, blocks_b, n_occupied)[whole]
    n_reference_bins = int(np.count_nonzero(reference > 0))
    insufficient = n_reference_bins < int(min_reference_bins)
    threshold = (np.inf if insufficient else
                 outer_fence_threshold(reference, min_effect_size=min_effect_size))
    # "Nothing here separates" is a claim about the field, and fixed grid bins
    # cannot quite make it: a fiducial that lands on a grid crossing is divided
    # among four of them and clears the fence in none, and returning "name
    # nothing" then threw away exactly that fiducial.  So a *compact* candidate
    # may answer for the field too -- one whose own neighbourhood holds a 2x2
    # block of shortlisted bins or less, which is all a source no wider than a
    # bin can cover at any grid phase.  Ordinary dense tissue fills all nine
    # and still cannot answer on the strength of where its window happened to
    # land.
    separates = bool(np.any(reference > threshold)) or any(
        item["pair_mass"] > threshold and item["footprint_bins"] <= _COMPACT_BINS
        for item in candidates)
    if not separates:
        threshold = np.inf
    # The fence's bulk is an upper quartile, so it breaks down once sources are
    # more than a quarter of the positive reference.  Say so instead of just
    # returning nothing.
    positive_reference = reference > 0
    shortlisted = np.zeros(n_occupied, dtype=bool)
    shortlisted[selected] = True
    shortlisted = shortlisted[whole]
    # Named for what it is observed to be: the shortlist is not the same as the
    # sources, since on a structured field most of it is ordinary dense tissue.
    # Context for a fence that declined, not a warning.
    reference_shortlist_fraction = (
        float(np.count_nonzero(shortlisted & positive_reference))
        / max(int(np.count_nonzero(positive_reference)), 1))
    accepted, needs_review, rejected = [], [], []
    for item in sorted(candidates, key=lambda c: c["pair_mass"], reverse=True):
        item["expected_false_bins"] = expected_false_bins(
            item["peak_bin_locs"], background_rate * item["peak_bin_exposure"],
            n_total_bins)
        by_chance = (max_expected_false_bins is not None
                     and item["expected_false_bins"] >= level)
        if insufficient:
            item["confidence"] = "rejected"
            item["reject_reason"] = "insufficient_reference"
            rejected.append(item)
        elif item["pair_mass"] >= threshold and not by_chance:
            # Spatially this is a fiducial.  Temporally it may be two brief
            # episodes far apart, which no cross-time pair count can tell from
            # a persistent source -- so that goes to review rather than being
            # removed automatically.
            if item["temporal_coverage"] >= _MIN_TEMPORAL_COVERAGE:
                item["confidence"] = "high"
                accepted.append(item)
            else:
                item["confidence"] = "needs_review"
                item["reject_reason"] = "temporal_coverage"
                needs_review.append(item)
        else:
            item["confidence"] = "rejected"
            item["reject_reason"] = "expected_by_chance" if by_chance else "below_fence"
            rejected.append(item)

    # Why nothing was named, when nothing was.  The fence is a threshold and
    # not a verdict, so it stands in almost every field; the case worth naming
    # is a fence that stood and that no candidate cleared.
    if insufficient:
        fence_undefined_reason = "insufficient_reference"
    elif accepted or needs_review:
        fence_undefined_reason = None
    elif reference_shortlist_fraction >= 0.25:
        fence_undefined_reason = "high_shortlist_fraction"
    else:
        fence_undefined_reason = "no_separation"

    centers_nm = (
        np.vstack([item["center_nm"] for item in accepted])
        if accepted else np.empty((0, 2), dtype=float)
    )
    if not return_report:
        return centers_nm

    report = {
        "accepted": accepted,
        "needs_review": needs_review,
        "rejected": rejected,
        "score_threshold": float(threshold),
        "min_effect_size": float(min_effect_size),
        "reference_shortlist_fraction": reference_shortlist_fraction,
        "fence_undefined_reason": fence_undefined_reason,
        "n_reference_bins": n_reference_bins,
        "min_reference_bins": int(min_reference_bins),
        "insufficient_reference": bool(insufficient),
        "field_bounds_nm": (x_min, x_max, y_min, y_max),
        "n_bins": int(n_total_bins),
        "n_locs_in_field": n_locs_in_field,
        "n_locs_outside_field": n_locs_outside_field,
        "background_rate_locs_per_bin": background_rate,
        "chance_occupancy": int(chance_occupancy),
        "first_pass_occupancy": int(first_pass_occupancy),
        "max_expected_false_bins": max_expected_false_bins,
        "bin_size_nm": bin_nm,
        "n_occupied_bins": int(n_occupied),
        "n_selected_bins": int(len(selected)),
        "n_components": int(len(components)),
        "time_block_frames": int(block_width),
        "n_time_blocks": int(n_time_blocks),
        "minimum_persistence": float(minimum_persistence),
        "effective_minimum_persistence": float(effective_persistence),
        "entropy_block_frames": int(entropy_width),
        "n_entropy_blocks": int(n_entropy_blocks),
        "pair_radius_nm": float(pair_radius_nm),
        "compute_fine_pair_mass": bool(compute_fine_pair_mass),
        "grid_offset_nm": (float(offset[0]), float(offset[1])),
        "merge_edges": bool(merge_edges),
    }
    return centers_nm, report


def _neighbour_positions(position, cell_ids, lookup, n_y, radius=1):
    """Compact indices of the occupied cells within ``radius`` cells."""
    cell_x, cell_y = divmod(int(cell_ids[position]), n_y)
    found = []
    for delta_x in range(-radius, radius + 1):
        for delta_y in range(-radius, radius + 1):
            if delta_x == 0 and delta_y == 0:
                continue
            if not 0 <= cell_y + delta_y < n_y:
                continue
            neighbour = lookup.get((cell_x + delta_x) * n_y + (cell_y + delta_y))
            if neighbour is not None:
                found.append(neighbour)
    return found


def _component_seeds(component, counts, cell_ids, lookup, n_y):
    """Local-maximum cells of one component, one seed each.

    A cell seeds when no neighbouring cell *in the same component* holds
    strictly more localizations.  Ties seed too: two adjacent cells holding a
    fiducial each have near-identical counts, and breaking the tie would drop
    one of them, while a genuine plateau is collapsed again by
    :func:`_fuse_coincident`.
    """
    members = set(int(position) for position in component)
    seeds = []
    for position in component:
        position = int(position)
        if all(counts[neighbour] <= counts[position]
               for neighbour in _neighbour_positions(position, cell_ids, lookup, n_y)
               if neighbour in members):
            seeds.append(position)
    return seeds or [int(component[int(np.argmax(counts[component]))])]


def _connected_components(selected, cell_ids, lookup, n_y):
    """8-connected components of the selected cells (the edge-merge step)."""
    selected_set = set(int(position) for position in selected)
    seen = set()
    components = []
    for seed in selected:
        seed = int(seed)
        if seed in seen:
            continue
        stack, component = [seed], []
        seen.add(seed)
        while stack:
            position = stack.pop()
            component.append(position)
            for neighbour in _neighbour_positions(position, cell_ids, lookup, n_y):
                if neighbour in selected_set and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _seed_halo(seed, cell_ids, lookup, n_y, order, cell_start):
    """Localizations near the seed, for the mean shift and the radius.

    Wide enough to hold the whole measuring disc and its background ring; the
    window the pair mass is scored over is only one bin.  Deliberately *not*
    the whole connected component: one component can cover most of a structured
    field, and scanning it per seed made the method quadratic in it.
    """
    positions = [int(seed)]
    positions.extend(_neighbour_positions(seed, cell_ids, lookup, n_y,
                                         radius=int(np.ceil(_BACKGROUND_ANNULUS))))
    chunks = [order[cell_start[c]:cell_start[c + 1]] for c in sorted(set(positions))]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _score_candidate(
    centre, window_idx, cloud_idx, xy, frames, blocks_a, blocks_b, bin_nm,
    pair_radius_nm, max_locs, compute_fine_pair_mass, n_bins, footprint_bins,
    peak_bin_locs, peak_bin_exposure, entropy_width, n_entropy_blocks,
    frame_min, frame_max, background_per_block,
    field_bounds, other_centres, grid_shift_nm,
):
    """Pair mass of one candidate, with time-uniform subsampling if needed."""
    window_frames = frames[window_idx]
    window_a = blocks_a[window_idx]
    window_b = blocks_b[window_idx]
    n_locs = int(len(window_idx))

    keep, stride = _time_uniform_subsample(window_frames, max_locs)
    sub_xy = xy[window_idx[keep]]
    scale = float(stride) ** 2

    # Coarse pair mass: a complete graph over the bin-sized window, analytic,
    # exact and O(n).  It never needs the subsample -- only the KD radius does.
    pair_mass = _analytic_cross_time_pairs(window_a, window_b)
    # Same quantity off the strided subsample, as a calibration of the
    # stride**2 estimator against a number we happen to know exactly.
    pair_mass_strided = scale * _analytic_cross_time_pairs(
        window_a[keep], window_b[keep])

    if compute_fine_pair_mass:
        pair_mass_fine = scale * _kd_cross_time_pairs(
            sub_xy, window_a[keep], window_b[keep], pair_radius_nm)
        uniform = pair_mass * (np.pi * pair_radius_nm ** 2) / (bin_nm ** 2)
        concentration = float(pair_mass_fine / uniform) if uniform > 0 else 0.0
    else:
        pair_mass_fine = concentration = float("nan")

    # How spread in time is this source's mass?  The window holds background as
    # well as the source -- for a faint one, mostly background -- so the
    # expected background is subtracted per block first, and the comparison is
    # against a *contiguous* source of the same span, so one attached for a
    # fifth of the acquisition scores like one present throughout.
    if n_locs:
        # The raw extent of everything in the window, background included: ten
        # stray background localizations stretched a true 4000-4099 burst to
        # (231, 7339), which is what a reviewer was being shown.
        first, last = int(window_frames.min()), int(window_frames.max())
        # Sparse in the block axis, for the same reason the pair mass is: a
        # fine persistence over a long acquisition makes the block range
        # enormous, and this window occupies a handful of it.
        present, occupied = np.unique(
            (window_frames - frame_min) // entropy_width, return_counts=True)
        counts = occupied.astype(np.float64) - background_per_block
        np.clip(counts, 0.0, None, out=counts)
        active = np.flatnonzero(counts > 0)
        source_frame_span = (first, last)
        if len(active) == 0 or counts.sum() <= 0:
            temporal_entropy = 0.0
            temporal_coverage = 0.0
        else:
            weights = counts[active] / counts.sum()
            temporal_entropy = float(
                -np.sum(weights * np.log(weights)) / np.log(max(n_entropy_blocks, 2)))
            # The span comes from where the source's *mass* is, not from any
            # block with a non-zero residual: subtracting a field-wide rate
            # leaves a little behind almost everywhere, which would stretch a
            # briefly-attached fiducial across the whole acquisition.
            cumulative = np.cumsum(counts) / counts.sum()
            low = int(present[int(np.searchsorted(cumulative, 0.01))])
            high = int(present[int(np.searchsorted(cumulative, 0.99))])
            span_blocks = max(high - low + 1, 1)
            expected = np.log(span_blocks) / np.log(max(n_entropy_blocks, 2))
            # Where the source's own mass is, in frames -- what a reviewer
            # should be shown, rather than the window's raw extent.  Resolved
            # to the block, and clamped: the last block runs past the end of
            # the acquisition whenever the width does not divide the span.
            source_frame_span = (
                int(np.clip(frame_min + low * entropy_width, frame_min, frame_max)),
                int(np.clip(frame_min + (high + 1) * entropy_width - 1,
                            frame_min, frame_max)))
            # span_blocks == 1 means the source is shorter than the finest
            # interval measured here, so there is no evidence of persistence at
            # all.  That is the conservative answer, not a free pass.
            temporal_coverage = (float(np.clip(temporal_entropy / expected, 0.0, 1.0))
                                 if expected > 0 else 0.0)
    else:
        first = last = 0
        source_frame_span = (0, 0)
        temporal_entropy = 0.0
        temporal_coverage = 0.0

    offsets = xy[cloud_idx] - centre
    order_r = np.argsort(np.linalg.norm(offsets, axis=1))
    radial = np.linalg.norm(offsets[order_r], axis=1)
    # Same convention as the sample grid in _observable_area: [0, 2*pi) with
    # the same origin, or the two sector indexings are rotated relative to each
    # other and the observable-area correction lands on the wrong sectors.
    angle = np.mod(np.arctan2(offsets[order_r, 1], offsets[order_r, 0]), 2.0 * np.pi)
    radius_nm, n_background_sectors = _background_corrected_radius(
        radial, angle, bin_nm, centre, field_bounds, others=other_centres)
    return {
        "center_nm": np.asarray(centre, dtype=float),
        "n_locs": float(n_locs),
        "peak_bin_locs": int(peak_bin_locs),
        "peak_bin_exposure": float(peak_bin_exposure),
        "pair_mass": float(pair_mass),
        "pair_mass_strided": float(pair_mass_strided),
        "pair_mass_fine": float(pair_mass_fine),
        "concentration": concentration,
        "subsample_stride": int(stride),
        "n_bins": int(n_bins),
        "footprint_bins": int(footprint_bins),
        "grid_shift_nm": float(grid_shift_nm),
        "radius_p99_nm": radius_nm,
        "n_background_sectors": int(n_background_sectors),
        "window_frame_span": (first, last),
        "source_frame_span": source_frame_span,
        "n_time_blocks_present": int(len(np.unique(window_a))),
        "temporal_entropy": temporal_entropy,
        "temporal_coverage": temporal_coverage,
    }


def _observable_area(centre, field_bounds, radius, outer, others=None,
                     exclusion=0.0, n_angles=_AREA_ANGLES, n_steps=_AREA_STEPS):
    """Observable disc area, and observable area of each ring sector.

    Measured on a polar sample grid rather than assumed to be the geometric
    area: the edge of the field and the territory of a neighbouring candidate
    both make part of the disc unobservable.
    """
    x_min, x_max, y_min, y_max = field_bounds
    angles = (np.arange(n_angles) + 0.5) * (2.0 * np.pi / n_angles)
    edges = np.linspace(0.0, outer, n_steps + 1)
    mid = (edges[:-1] + edges[1:]) / 2.0
    sample_x = centre[0] + mid[:, None] * np.cos(angles)[None, :]
    sample_y = centre[1] + mid[:, None] * np.sin(angles)[None, :]
    inside = ((sample_x >= x_min) & (sample_x < x_max)
              & (sample_y >= y_min) & (sample_y < y_max))
    if others is not None and len(others) and exclusion > 0:
        for other in others:
            inside &= ((sample_x - other[0]) ** 2
                       + (sample_y - other[1]) ** 2) > exclusion ** 2
    cell = inside * (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / n_angles)[:, None]

    disc_area = float(cell[mid < radius].sum())
    sector_of_angle = (angles * _BACKGROUND_SECTORS
                       / (2.0 * np.pi)).astype(np.int64) % _BACKGROUND_SECTORS
    sector_area = np.bincount(sector_of_angle,
                              weights=cell[mid >= radius].sum(axis=0),
                              minlength=_BACKGROUND_SECTORS)
    return disc_area, sector_area


def _background_corrected_radius(radial, angle, bin_nm, centre, field_bounds,
                                 others=None):
    """Removal radius of the source, with the local background taken out.

    The raw percentile of everything in the measuring disc is not the source's
    radius: background grows with area, so for anything but a very bright
    source the outer part dominates and the percentile lands on the boundary.
    So the local density is estimated and used to work out how many
    localizations in the disc are the source; because the source is compact and
    ``radial`` is sorted, that many innermost distances are taken to be it.

    The density is the median over angular sectors of the ring, each normalised
    by its own *observable* area.  The median stops a resolved neighbour from
    inflating it, and the area normalisation stops a sector that lies mostly
    outside the field from reading as empty.  Neighbouring candidates are
    masked out of both the counts and the area: with a two-bin disc a resolved
    neighbour can fall inside it, and it is another source, not this one's
    background and not its cloud.

    Notes
    -----
    "The source is the innermost n" assumes its density falls off with radius,
    which holds for a drift-smeared point source and is what makes this stable;
    a cumulative-excess profile needs no such assumption but was measured worse
    twice.  The cost is precision on very faint sources, where the source count
    is known only to a handful and the percentile sits where background begins.
    That errs towards over-removal, deliberately -- the failure that matters is
    leaving a fiducial partly in the data.
    """
    if len(radial) == 0:
        return 0.0, 0
    radius = _SOURCE_DISC * float(bin_nm)
    outer = _BACKGROUND_ANNULUS * float(bin_nm)
    exclusion = float(bin_nm)

    # A neighbouring candidate is another source, not this one's background and
    # not this one's cloud.  Now that the measuring disc is two bins wide -- the
    # resolution limit -- a resolved neighbour can fall inside it, so its
    # territory is masked out of the counts here and out of the area below.
    if others is not None and len(others):
        keep = np.ones(len(radial), dtype=bool)
        for other in others:
            offset = np.asarray(other, dtype=float) - np.asarray(centre, dtype=float)
            distance = np.hypot(radial * np.cos(angle) - offset[0],
                                radial * np.sin(angle) - offset[1])
            keep &= distance > exclusion
        radial, angle = radial[keep], angle[keep]
        if len(radial) == 0:
            return 0.0, 0

    inside = radial[radial < radius]
    if len(inside) == 0:
        return 0.0, 0

    disc_area, sector_area = _observable_area(
        centre, field_bounds, radius, outer, others=others, exclusion=exclusion)
    in_ring = radial >= radius
    counted = np.bincount(
        (angle[in_ring] * _BACKGROUND_SECTORS / (2.0 * np.pi)).astype(np.int64)
        % _BACKGROUND_SECTORS, minlength=_BACKGROUND_SECTORS)
    usable = (sector_area > _MIN_OBSERVABLE_SECTOR * sector_area.max()
              if sector_area.max() > 0
              else np.zeros(_BACKGROUND_SECTORS, dtype=bool))
    n_sectors = int(np.count_nonzero(usable))
    density = (float(np.median(counted[usable] / sector_area[usable]))
               if n_sectors else 0.0)

    n_source = int(round(len(inside) - density * disc_area))
    n_source = int(np.clip(n_source, 1, len(inside)))
    return float(np.percentile(inside[:n_source], _CLOUD_PERCENTILE)), n_sectors


def _fuse_coincident(candidates, bin_nm):
    """Keep the stronger of any two candidates within one bin of each other.

    Greedy, strongest first, through a KD-tree: a structured field can shortlist
    thousands of candidates and the pairwise form is quadratic in that.
    """
    if len(candidates) < 2:
        return list(candidates)
    order = sorted(range(len(candidates)),
                   key=lambda i: candidates[i]["pair_mass"], reverse=True)
    centres = np.vstack([item["center_nm"] for item in candidates])
    tree = cKDTree(centres)
    neighbours = tree.query_ball_point(centres, r=float(bin_nm))
    suppressed = np.zeros(len(candidates), dtype=bool)
    kept = []
    for index in order:
        if suppressed[index]:
            continue
        kept.append(candidates[index])
        for other in neighbours[index]:
            if other != index:
                suppressed[other] = True
    return kept


def fiducial_mask(dataset, centers_nm, radius_nm):
    """Return a boolean mask of localizations near a fiducial centre.

    Parameters
    ----------
    dataset : (N, >=2) array_like
        Localizations; only the first two columns are read.  Rows with a
        non-finite x or y are simply never masked, matching
        :func:`find_fiducials`, which drops them.
    centers_nm : (M, 2) array_like
        Fiducial centres, e.g. from :func:`find_fiducials`.  An empty array
        yields an all-``False`` mask.
    radius_nm : float or (M,) array_like
        Removal radius, either one value for all centres or one per centre.
        Per-centre is the better choice: each accepted candidate reports its
        own measured cloud extent as ``radius_p99_nm``.  Zero is allowed and
        means an exact coordinate match -- a perfectly stationary or heavily
        quantized source can legitimately measure zero, and masking it should
        not be an error.

    Returns
    -------
    ndarray of bool
        ``True`` where the localization belongs to a fiducial neighbourhood.

    Raises
    ------
    ValueError
        If ``radius_nm`` is neither a scalar nor one value per centre, or if
        any radius is negative or not finite.

    Examples
    --------
    >>> import numpy as np
    >>> locs = np.array([[0., 0., 0., 0.], [500., 0., 0., 1.], [40., 0., 0., 2.]])
    >>> fiducial_mask(locs, [[0., 0.]], 50.0)
    array([ True, False,  True])
    >>> fiducial_mask(locs, [[0., 0.], [500., 0.]], [10.0, 60.0])
    array([ True,  True, False])
    """
    dataset = np.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[1] < 2:
        raise ValueError("dataset must have shape (N, >=2).")
    xy = np.column_stack([dataset[:, 0].astype(np.float64),
                          dataset[:, 1].astype(np.float64)])
    centers_nm = np.asarray(centers_nm, dtype=float).reshape(-1, 2)
    mask = np.zeros(len(dataset), dtype=bool)
    if len(centers_nm) == 0 or len(dataset) == 0:
        return mask

    radii = np.asarray(radius_nm, dtype=float)
    if radii.ndim == 0:
        radii = np.full(len(centers_nm), float(radii))
    elif radii.shape != (len(centers_nm),):
        raise ValueError(
            "radius_nm must be a scalar or one value per centre; got shape %s "
            "for %d centres." % (radii.shape, len(centers_nm)))
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("every radius must be finite and not negative.")

    finite = np.flatnonzero(np.isfinite(xy).all(axis=1))
    if len(finite) == 0:
        return mask
    tree = cKDTree(xy[finite])
    for index in tree.query_ball_point(centers_nm, r=radii):
        mask[finite[np.asarray(index, dtype=np.int64)]] = True
    return mask
