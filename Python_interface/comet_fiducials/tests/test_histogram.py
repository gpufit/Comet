"""Fast synthetic tests for :mod:`comet_fiducials`.

No external data: everything here runs in a few seconds.  The real-data
validation against seventeen hand-reviewed bead centres is described in
``docs/fiducials.md``.
"""
import numpy as np
import pytest

from comet_fiducials.histogram import _SOURCE_DISC
from comet_fiducials import (expected_false_bins, fiducial_mask,
                             field_bounds_from_data, find_fiducials,
                             outer_fence_threshold)

MAX_DRIFT_NM = 250.0
FIELD = (-250.0, 20_250.0, -250.0, 20_250.0)


def _dataset(bead_centers, seed=0, n_frames=8_000, n_background=60_000, every=1):
    """Uniform background plus point sources, all sharing one drift path."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 1, (n_frames, 2)), axis=0)
    walk -= walk[0]
    walk *= (MAX_DRIFT_NM * 0.45) / np.abs(walk).max()

    frames = rng.integers(0, n_frames, n_background)
    parts = [np.column_stack([
        rng.uniform(0, 20_000, n_background) + walk[frames, 0],
        rng.uniform(0, 20_000, n_background) + walk[frames, 1],
        np.zeros(n_background), frames,
    ])]
    truth = []
    for centre in np.asarray(bead_centers, dtype=float).reshape(-1, 2):
        bead_frames = np.arange(0, n_frames, every)
        position = centre + walk[bead_frames] + rng.normal(0, 8.0, (len(bead_frames), 2))
        parts.append(np.column_stack([
            position, np.zeros(len(bead_frames)), bead_frames]))
        truth.append(centre + walk[bead_frames].mean(axis=0))
    return np.vstack(parts), np.asarray(truth)


def _uniform(n=80_000, side=20_000.0, n_frames=8_000, seed=3):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0, side, n), rng.uniform(0, side, n), np.zeros(n),
        rng.integers(0, n_frames, n).astype(float),
    ])


def _errors(found, truth):
    assert len(found) == len(truth), "expected %d beads, got %d" % (len(truth), len(found))
    return np.min(np.linalg.norm(truth[:, None, :] - found[None, :, :], axis=2), axis=1)


# --------------------------------------------------------------------------
# core behaviour
# --------------------------------------------------------------------------
def test_finds_every_bead_and_nothing_else():
    dataset, truth = _dataset([[5_000, 5_000], [12_345.6, 7_654.3], [15_000, 15_000]])
    found = find_fiducials(dataset, MAX_DRIFT_NM, FIELD, return_report=False)
    assert np.all(_errors(found, truth) < 20.0)


def test_result_does_not_depend_on_the_grid_phase():
    """The edge merge plus box mean shift must keep the grid out of the answer.

    The accepted set has to be identical at every phase; the centres are
    allowed to move a little, because which bin is a local maximum is itself a
    function of phase.
    """
    dataset, truth = _dataset([[5_000, 5_000], [10_000, 7_500], [12_625, 9_375]])
    for offset in (0.0, 31.25, 62.5, 125.0, 187.5):
        found = find_fiducials(
            dataset, MAX_DRIFT_NM, FIELD, grid_offset_nm=(offset, offset),
            return_report=False)
        found = found[np.lexsort((found[:, 1], found[:, 0]))]
        assert np.all(_errors(found, truth) < 25.0)


def test_whole_bin_offsets_are_the_same_grid():
    """Offsets differing by whole bins, and negative ones, must not alias."""
    dataset, _ = _dataset([[5_000, 5_000]])
    baseline = find_fiducials(dataset, MAX_DRIFT_NM, FIELD, return_report=False)
    for offset in (MAX_DRIFT_NM, -MAX_DRIFT_NM, 3 * MAX_DRIFT_NM):
        shifted = find_fiducials(
            dataset, MAX_DRIFT_NM, FIELD, grid_offset_nm=(offset, offset),
            return_report=False)
        assert np.allclose(shifted, baseline)
    # A negative fractional offset must not fold two cells onto one key.
    three_cells = np.array([[0., 0., 0., 0.], [260., 0., 0., 1.], [0., 260., 0., 2.]])
    for offset in ((0.0, 0.0), (-125.0, -125.0), (125.0, 125.0)):
        _, report = find_fiducials(three_cells, MAX_DRIFT_NM, FIELD,
                                   grid_offset_nm=offset, min_reference_bins=0)
        assert report["n_occupied_bins"] == 3


def test_a_uniform_field_names_nothing():
    dataset = _uniform()
    for max_drift in (100.0, 250.0, 760.0):
        found = find_fiducials(dataset, max_drift, FIELD, return_report=False)
        assert len(found) == 0, "max_drift=%g named %d" % (max_drift, len(found))


def test_partially_attached_beads_are_still_found():
    """Attaching late or detaching early must not be confused with a burst."""
    rng = np.random.default_rng(9)
    n_frames = 20_000
    parts, truth = [], []
    for centre, first, last in (([5_000.0, 5_000.0], 0, 4_000),
                                ([15_000.0, 15_000.0], 16_000, n_frames),
                                ([10_000.0, 15_000.0], 0, n_frames)):
        frames = np.arange(first, last)
        parts.append(np.column_stack([
            rng.normal(centre[0], 8.0, len(frames)), rng.normal(centre[1], 8.0, len(frames)),
            np.zeros(len(frames)), frames.astype(float)]))
        truth.append(centre)
    background_frames = rng.integers(0, n_frames, 300_000)
    parts.append(np.column_stack([
        rng.uniform(0, 20_000, 300_000), rng.uniform(0, 20_000, 300_000),
        np.zeros(300_000), background_frames.astype(float)]))
    found, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD)
    assert len(found) == 3
    assert np.all(_errors(found, np.asarray(truth)) < 30.0)
    # Spread over a real fraction of the acquisition, not one episode.
    assert min(item["temporal_entropy"] for item in report["accepted"]) > 0.5


# --------------------------------------------------------------------------
# the statistical model
# --------------------------------------------------------------------------
def test_a_coordinate_outlier_does_not_change_the_answer():
    """The field of view must not be inferred from raw coordinate extrema.

    One far localization would otherwise inflate the bin count by orders of
    magnitude, drive the background rate to zero, and make every ordinary
    fluctuation significant.
    """
    dataset = _uniform()
    outlier = np.vstack([dataset, [[5e9, 5e9, 0.0, 0.0]]])
    for max_drift in (100.0, 250.0):
        clean, clean_report = find_fiducials(dataset, max_drift, FIELD)
        dirty, dirty_report = find_fiducials(outlier, max_drift, FIELD)
        assert len(clean) == len(dirty) == 0
        assert dirty_report["n_bins"] == clean_report["n_bins"]
        assert dirty_report["background_rate_locs_per_bin"] == pytest.approx(
            clean_report["background_rate_locs_per_bin"], rel=1e-3)


def test_explicit_field_bounds_are_honoured_and_validated():
    dataset = _uniform()
    _, report = find_fiducials(dataset, MAX_DRIFT_NM, (0.0, 20_000.0, 0.0, 20_000.0))
    assert report["field_bounds_nm"] == (0.0, 20_000.0, 0.0, 20_000.0)
    assert report["n_bins"] == 80 * 80
    with pytest.raises(ValueError, match="four finite numbers"):
        find_fiducials(dataset, MAX_DRIFT_NM, (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="x_max > x_min"):
        find_fiducials(dataset, MAX_DRIFT_NM, (10.0, 0.0, 0.0, 10.0))
    with pytest.raises(ValueError, match="field_bounds_nm is required"):
        find_fiducials(dataset, MAX_DRIFT_NM, None)


def test_a_bead_free_field_inside_one_bin_is_not_a_fiducial():
    """An outlier test needs a sample; one bin is not one."""
    rng = np.random.default_rng(0)
    n = 20_000
    dataset = np.column_stack([
        rng.uniform(0, 200, n), rng.uniform(0, 200, n), np.zeros(n),
        rng.integers(0, 8_000, n).astype(float)])
    found, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD)
    assert len(found) == 0
    assert report["insufficient_reference"]
    assert report["n_reference_bins"] < report["min_reference_bins"]
    assert all(item["reject_reason"] == "insufficient_reference"
               for item in report["rejected"])


def test_a_contiguous_dense_region_is_not_a_field_of_fiducials():
    """A bead-free sparse field whose bin pair masses take only a few values.

    This is the case the effect-size floor exists for, and the one that fixes
    its value: with the floor below 25 it names 34 fiducials here.  The dense
    region sits at 19x the bulk pair mass, so the margin is only 1.3x -- see the
    measured curve beside ``_MIN_EFFECT_SIZE``.
    """
    rng = np.random.default_rng(0)
    parts = []
    for cell_x in range(40):
        for cell_y in range(40):
            dense = 16 <= cell_x < 24 and 16 <= cell_y < 24
            n = 20 if dense else (5 if (cell_x + cell_y) % 2 else 4)
            parts.append(np.column_stack([
                rng.uniform(cell_x * 250 + 20, (cell_x + 1) * 250 - 20, n),
                rng.uniform(cell_y * 250 + 20, (cell_y + 1) * 250 - 20, n),
                np.zeros(n), np.linspace(0, 7_999, n)]))
    dataset = np.vstack(parts)
    field = (0.0, 10_000.0, 0.0, 10_000.0)
    assert len(find_fiducials(dataset, MAX_DRIFT_NM, field, return_report=False)) == 0
    # ...and it is the floor that refuses it, not anything else.
    lax = find_fiducials(dataset, MAX_DRIFT_NM, field, min_effect_size=5.0,
                         return_report=False)
    assert len(lax) > 10


def test_the_fence_is_continuous_across_a_flat_bulk():
    """An earlier version branched on `q1 == q3` exactly and flipped across it."""
    reference = np.concatenate([np.full(1_545, 10.0), np.full(55, 105.0)])
    jittered = reference.copy()
    jittered[:1_545] += np.linspace(0.0, 1e-12, 1_545)
    assert outer_fence_threshold(reference) == pytest.approx(
        outer_fence_threshold(jittered), rel=1e-9)
    # A second mode ten times the bulk is a mode and not a tail: the fence
    # stands above all of it rather than cutting through it.
    assert outer_fence_threshold(reference) > reference.max()


def test_a_broad_dense_mode_is_not_a_field_of_fiducials():
    """A bead-free field with two density modes must not name the upper one.

    This is what the old largest-gap fallback got wrong: it happily split a
    reference into two modes and called the whole upper mode outliers.
    """
    rng = np.random.default_rng(7)
    parts = []
    for cell_x in range(40):
        for cell_y in range(40):
            dense = (cell_x * 7 + cell_y * 3) % 29 == 0
            n = int(rng.poisson(15 if dense else 5))
            if n == 0:
                continue
            parts.append(np.column_stack([
                rng.uniform(cell_x * 250 + 20, (cell_x + 1) * 250 - 20, n),
                rng.uniform(cell_y * 250 + 20, (cell_y + 1) * 250 - 20, n),
                np.zeros(n), np.linspace(0, 7_999, n)]))
    found, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD)
    assert report["n_selected_bins"] > 10, "the dense mode should reach the shortlist"
    assert len(found) == 0, "but nothing should be named a fiducial"


def test_many_beads_are_not_capped_by_the_shortlist():
    """The proposal cut is derived from the density, not a fixed percentile.

    A percentile shortlist can only ever propose that fraction of the field,
    so twenty beads among a thousand bins could not all be proposed.
    """
    rng = np.random.default_rng(11)
    parts, truth = [], []
    for index in range(1_000):
        cell_x, cell_y = divmod(index, 32)
        n = int(rng.poisson(5))
        if n == 0:
            continue
        parts.append(np.column_stack([
            rng.uniform(cell_x * 250 + 20, cell_x * 250 + 230, n),
            rng.uniform(cell_y * 250 + 20, cell_y * 250 + 230, n),
            np.zeros(n), np.linspace(0, 7_999, n)]))
    for bead in range(20):
        cell_x, cell_y = divmod(bead * 37, 32)
        n = 2_000 + bead * 200
        centre = (cell_x * 250 + 125.0, cell_y * 250 + 125.0)
        truth.append(centre)
        parts.append(np.column_stack([
            rng.normal(centre[0], 8.0, n), rng.normal(centre[1], 8.0, n),
            np.zeros(n), np.linspace(0, 7_999, n)]))
    found, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD)
    assert report["n_selected_bins"] >= 20
    assert len(found) == 20
    assert np.all(_errors(found, np.asarray(truth)) < 30.0)


def test_chance_test_keeps_a_faint_source_on_a_sparse_field():
    """lambda and the trial count must describe the same all-bin population."""
    rng = np.random.default_rng(3)
    n_background = 7_500
    side = np.sqrt(150_000) * MAX_DRIFT_NM
    background = np.column_stack([
        rng.uniform(0, side, n_background), rng.uniform(0, side, n_background),
        np.zeros(n_background), rng.integers(0, 8_000, n_background).astype(float)])
    source = np.column_stack([
        side / 2 + rng.normal(0, 8.0, 5), side / 2 + rng.normal(0, 8.0, 5),
        np.zeros(5), np.linspace(0, 7_999, 5)])
    dataset = np.vstack([background, source])
    _, report = find_fiducials(dataset, MAX_DRIFT_NM, (0.0, side, 0.0, side))
    true_rate = len(dataset) / report["n_bins"]
    assert report["background_rate_locs_per_bin"] == pytest.approx(true_rate, rel=0.25)
    candidate = min(report["accepted"] + report["needs_review"] + report["rejected"],
                    key=lambda c: np.linalg.norm(c["center_nm"] - side / 2))
    assert candidate["expected_false_bins"] < 0.01
    assert candidate.get("reject_reason") != "expected_by_chance"


# --------------------------------------------------------------------------
# the temporal statistic
# --------------------------------------------------------------------------
def test_burst_score_does_not_depend_on_the_temporal_grid_phase():
    """Two half-offset time grids, smaller kept: a burst cannot be promoted."""
    def burst_and_bead(first_frame):
        rng = np.random.default_rng(1)
        n_frames = 8_000
        frames = np.repeat(np.arange(first_frame, first_frame + 200), 100)
        burst = np.column_stack([
            rng.normal(5_000, 8.0, len(frames)), rng.normal(5_000, 8.0, len(frames)),
            np.zeros(len(frames)), frames.astype(float)])
        bead_frames = np.arange(n_frames)
        bead = np.column_stack([
            rng.normal(15_000, 8.0, n_frames), rng.normal(15_000, 8.0, n_frames),
            np.zeros(n_frames), bead_frames.astype(float)])
        background_frames = rng.integers(0, n_frames, 40_000)
        background = np.column_stack([
            rng.uniform(0, 20_000, 40_000), rng.uniform(0, 20_000, 40_000),
            np.zeros(40_000), background_frames.astype(float)])
        _, report = find_fiducials(np.vstack([burst, bead, background]),
                                   MAX_DRIFT_NM, FIELD)
        everything = report["accepted"] + report["needs_review"] + report["rejected"]
        pick = lambda target: min(  # noqa: E731
            everything, key=lambda c: np.linalg.norm(c["center_nm"] - target))
        return pick([5_000, 5_000]), pick([15_000, 15_000])

    inside, _ = burst_and_bead(4_000)
    straddling, bead = burst_and_bead(4_150)
    assert inside["n_locs"] > 15_000 and straddling["n_locs"] > 15_000
    assert straddling["pair_mass"] == pytest.approx(inside["pair_mass"], rel=0.05)
    assert bead["pair_mass"] > 100.0 * straddling["pair_mass"]


def test_minimum_persistence_sets_how_brief_a_burst_is():
    """The timescale is one named policy, and it does what it says.

    A burst shorter than `minimum_persistence` of the acquisition is
    suppressed; relax the policy below the burst's own length and it is not.
    Documented rather than derived -- hence a test that pins the meaning.
    """
    rng = np.random.default_rng(1)
    n_frames = 8_000
    frames = np.repeat(np.arange(4_000, 4_200), 100)
    burst = np.column_stack([
        rng.normal(5_000, 8.0, len(frames)), rng.normal(5_000, 8.0, len(frames)),
        np.zeros(len(frames)), frames.astype(float)])
    background_frames = rng.integers(0, n_frames, 40_000)
    background = np.column_stack([
        rng.uniform(0, 20_000, 40_000), rng.uniform(0, 20_000, 40_000),
        np.zeros(40_000), background_frames.astype(float)])
    dataset = np.vstack([burst, background])

    def burst_pair_mass(minimum_persistence):
        _, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD,
                                   minimum_persistence=minimum_persistence)
        return min(report["accepted"] + report["needs_review"] + report["rejected"],
                   key=lambda c: np.linalg.norm(c["center_nm"] - 5_000))["pair_mass"]

    # Blocks are twice the policy wide, so a *smaller* minimum_persistence
    # means narrower blocks and less suppression.
    burst_fraction = 200.0 / n_frames                  # 1/40 of the acquisition
    assert burst_pair_mass(1.0 / 64.0) < 1e6              # 1/64 < 1/40: suppressed
    assert burst_pair_mass(1.0 / 256.0) > 1e7             # 1/256 well under: not
    assert 1.0 / 256.0 < burst_fraction < 1.0 / 16.0

    # The value asked for is the value used, not one rounded into a block count.
    for requested in (0.4, 0.3, 0.2, 0.1, 1.0 / 64.0):
        _, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD,
                                   minimum_persistence=requested)
        assert report["effective_minimum_persistence"] == pytest.approx(
            requested, rel=0.02)
    for bad in (0.9, 0.5, 0.0, -0.1):
        with pytest.raises(ValueError, match="minimum_persistence"):
            find_fiducials(dataset, MAX_DRIFT_NM, FIELD, minimum_persistence=bad)
    # 0.5 is excluded because it leaves one acquisition-wide block, and then
    # one of the two time grids has no cross-block pairs at all and every
    # pair mass collapses to zero.  Just below it a real bead is still found --
    # assert the behaviour, not only the reported value.
    with_bead, truth = _dataset([[5_000, 5_000]])
    for persistence in (0.49, 0.25, 1.0 / 64.0):
        found = find_fiducials(with_bead, MAX_DRIFT_NM, FIELD,
                               minimum_persistence=persistence, return_report=False)
        assert len(found) == 1, "persistence %.3f found %d" % (persistence, len(found))
        assert np.all(_errors(found, truth) < 25.0)


def test_duty_cycle_separates_a_flickering_source_from_a_bead():
    """Cross-time pairs cannot tell two far-apart bursts from a bead.

    The pair mass measures temporal spread, so a source that appears twice at
    opposite ends of the movie scores like a persistent one.  `duty_cycle` is
    what makes that visible in the report.
    """
    rng = np.random.default_rng(1)
    n_frames = 8_000
    flicker_frames = np.concatenate([np.repeat(np.arange(0, 100), 100),
                                     np.repeat(np.arange(7_900, 8_000), 100)])
    flicker = np.column_stack([
        rng.normal(5_000, 8.0, len(flicker_frames)),
        rng.normal(5_000, 8.0, len(flicker_frames)),
        np.zeros(len(flicker_frames)), flicker_frames.astype(float)])
    bead_frames = np.arange(n_frames)
    bead = np.column_stack([
        rng.normal(15_000, 8.0, n_frames), rng.normal(15_000, 8.0, n_frames),
        np.zeros(n_frames), bead_frames.astype(float)])
    background_frames = rng.integers(0, n_frames, 40_000)
    background = np.column_stack([
        rng.uniform(0, 20_000, 40_000), rng.uniform(0, 20_000, 40_000),
        np.zeros(40_000), background_frames.astype(float)])
    _, report = find_fiducials(np.vstack([flicker, bead, background]),
                               MAX_DRIFT_NM, FIELD)
    everything = report["accepted"] + report["needs_review"] + report["rejected"]
    pick = lambda t: min(  # noqa: E731
        everything, key=lambda c: np.linalg.norm(c["center_nm"] - t))
    flickering, persistent = pick([5_000, 5_000]), pick([15_000, 15_000])
    # The pair mass really cannot tell them apart...
    assert flickering["pair_mass"] > persistent["pair_mass"]
    # ...but the temporal entropy can, and it is in the report.
    assert flickering["temporal_entropy"] < 0.35
    assert persistent["temporal_entropy"] > 0.95


def test_time_uniform_subsample_reproduces_the_exact_pair_mass():
    dataset, _ = _dataset([[5_000, 5_000]], n_background=20_000, n_frames=40_000)
    _, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD, max_locs_per_candidate=2_000)
    bead = report["accepted"][0]
    assert bead["subsample_stride"] > 1
    relative = abs(bead["pair_mass_strided"] - bead["pair_mass"]) / bead["pair_mass"]
    assert relative < 0.01, "stride**2 estimator off by %.3f%%" % (100 * relative)


# --------------------------------------------------------------------------
# resolution, memory, plumbing
# --------------------------------------------------------------------------
def test_two_beads_merge_below_the_bin_and_resolve_above_twice_it():
    """The resolution limit, measured rather than asserted.

    Below about one bin two beads always come back as one centre; at two bins
    and beyond they always come back as two.  In between it depends on where
    the grid lines fall relative to the pair, because a bead split across two
    cells can be swallowed by a stronger neighbour.
    """
    def two_beads(separation_nm, grid_offset_nm=(0.0, 0.0)):
        rng = np.random.default_rng(4)
        n_frames = 8_000
        parts = []
        for centre_x in (5_000.0, 5_000.0 + separation_nm):
            parts.append(np.column_stack([
                rng.normal(centre_x, 8.0, n_frames), rng.normal(5_000, 8.0, n_frames),
                np.zeros(n_frames), np.arange(n_frames).astype(float)]))
        background_frames = rng.integers(0, n_frames, 40_000)
        parts.append(np.column_stack([
            rng.uniform(0, 20_000, 40_000), rng.uniform(0, 20_000, 40_000),
            np.zeros(40_000), background_frames.astype(float)]))
        return find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD,
                              grid_offset_nm=grid_offset_nm, return_report=False)

    for separation in (0.6, 1.0, 1.2):
        assert len(two_beads(separation * MAX_DRIFT_NM)) == 1, (
            "%.1f x bin should merge" % separation)
    for separation in (2.0, 3.0):
        for offset in (0.0, 62.5, 125.0, 187.5):
            found = two_beads(separation * MAX_DRIFT_NM, (offset, offset))
            assert len(found) == 2, (
                "%.1f x bin at phase %.1f gave %d" % (separation, offset, len(found)))
            assert abs(np.ptp(found[:, 0]) - separation * MAX_DRIFT_NM) < 30.0


def test_memory_does_not_scale_with_the_time_grid():
    """The fence reference must stay sparse in the block axis.

    Sixteen times as many time blocks would cost sixteen times the memory in
    the dense form.  The invariant is that the peak barely moves.

    ``tracemalloc`` is used rather than ``ru_maxrss``: the latter is bytes on
    macOS and KiB on Linux, and reports a process-lifetime maximum that any
    earlier test can already have set.
    """
    import tracemalloc
    rng = np.random.default_rng(0)
    n = 400_000
    dataset = np.column_stack([
        rng.uniform(0, 2e5, n), rng.uniform(0, 2e5, n), np.zeros(n),
        rng.integers(0, 8_000, n).astype(float)])

    wide_field = (0.0, 2e5, 0.0, 2e5)

    def peak_bytes(minimum_persistence):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            _, report = find_fiducials(dataset, MAX_DRIFT_NM, wide_field,
                                       minimum_persistence=minimum_persistence)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak, report

    coarse, coarse_report = peak_bytes(1.0 / 64.0)
    fine, fine_report = peak_bytes(1.0 / 1024.0)
    assert fine_report["n_time_blocks"] > 10 * coarse_report["n_time_blocks"]
    dense_growth = (fine_report["n_time_blocks"] - coarse_report["n_time_blocks"]) \
        * coarse_report["n_occupied_bins"] * 8
    assert dense_growth > 1e9, "the dense form would have grown by gigabytes"
    assert fine - coarse < 0.25 * coarse, (
        "peak grew from %.0f MB to %.0f MB with the block count"
        % (coarse / 1e6, fine / 1e6))


def test_a_single_coordinate_outlier_does_not_allocate_the_bounding_box():
    dataset = _uniform(n=20_000)
    dataset[0, 0] = 5e9
    found, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD)
    assert report["n_occupied_bins"] < 20_000
    assert len(found) == 0

    # Explicit bounds that exclude everything are an error, not an empty answer.
    with pytest.raises(ValueError, match="inside field_bounds_nm"):
        find_fiducials(dataset, MAX_DRIFT_NM, (1e12, 2e12, 1e12, 2e12))


def test_z_is_ignored_not_required():
    """Z is documented as unused, so a missing z must not cost the row."""
    dataset, _ = _dataset([[5_000, 5_000]])
    with_nan_z = dataset.copy()
    with_nan_z[::3, 2] = np.nan
    assert np.array_equal(find_fiducials(dataset, MAX_DRIFT_NM, FIELD, return_report=False),
                          find_fiducials(with_nan_z, MAX_DRIFT_NM, FIELD, return_report=False))


def test_fence_declines_rather_than_splitting_a_homogeneous_sample():
    """The fence is a threshold, not a verdict.

    With fewer than two positive scores there is nothing to place one against
    and it is ``inf``; otherwise it is always a number, and a homogeneous
    sample is declined by that number standing above the whole sample rather
    than by a verdict returned in its place.  Deciding "nothing separates"
    here, against the reference, is what discarded a fiducial that had landed
    on a grid crossing and so cleared no single bin of its own.
    """
    assert outer_fence_threshold([]) == np.inf
    assert outer_fence_threshold([1_000.0]) == np.inf
    flat = np.full(50, 1_000.0)
    assert outer_fence_threshold(flat) > flat.max()
    assert np.isfinite(outer_fence_threshold([1, 1, 2, 2, 3, 1000]))
    assert outer_fence_threshold([1, 1, 2, 2, 3, 1000]) < 1000


def test_fence_still_names_a_lone_outlier_in_a_flat_reference():
    """A field of identical bins with one enormous source in it.

    The quartiles carry no information, so the fence falls back to a discrete
    upper tail no larger than the budget.  A flat *bimodal* reference has no
    such tail and is still declined.
    """
    lone = np.concatenate([np.full(400, 10.0), [1e6]])
    threshold = outer_fence_threshold(lone)
    assert 10.0 < threshold < 1e6, "the threshold must separate, not equal, the tail"
    # A budget is not a cap: many equally strong sources are all named.
    many = np.concatenate([np.full(400, 10.0), np.full(12, 1e6)])
    assert outer_fence_threshold(many) == pytest.approx(threshold)
    # A second mode only ten times the bulk is a mode, not a tail: the fence
    # stands above it, so nothing scored against it is named.
    bimodal = np.concatenate([np.full(1_545, 10.0), np.full(55, 105.0)])
    assert outer_fence_threshold(bimodal) > bimodal.max()


def test_enormous_sources_in_a_flat_field_are_all_found():
    """However many there are: the degenerate branch is not a count cap."""
    def flat_field_with(n_sources):
        rng = np.random.default_rng(1)
        parts = []
        for index in range(400):
            cell_x, cell_y = divmod(index, 20)
            parts.append(np.column_stack([
                rng.uniform(cell_x * 250 + 20, cell_x * 250 + 230, 6),
                rng.uniform(cell_y * 250 + 20, cell_y * 250 + 230, 6),
                np.zeros(6), np.linspace(0, 7_999, 6)]))
        for source in range(n_sources):
            centre_x = 2_500.0 + source * 1_500.0
            parts.append(np.column_stack([
                rng.normal(centre_x, 8.0, 30_000), rng.normal(2_500, 8.0, 30_000),
                np.zeros(30_000), np.linspace(0, 7_999, 30_000)]))
        return find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD, return_report=False)

    for n_sources in (1, 2, 3):
        assert len(flat_field_with(n_sources)) == n_sources


def test_expected_false_bins_matches_the_stated_model():
    assert expected_false_bins(0, 2.0, 100) == 0.0
    assert expected_false_bins(5, 0.0, 100) == 0.0
    assert expected_false_bins(9, 2.0, 40_000) > 1.0
    assert expected_false_bins(30, 2.0, 40_000) < 1e-6


def test_removal_mask_accepts_one_radius_per_centre():
    dataset, _ = _dataset([[5_000, 5_000], [15_000, 15_000]])
    found, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD)
    radii = np.asarray([item["radius_p99_nm"] for item in report["accepted"]])
    per_bead = fiducial_mask(dataset, found, radii)
    flat = fiducial_mask(dataset, found, MAX_DRIFT_NM)
    assert per_bead.sum() <= flat.sum()
    assert per_bead.sum() > 10_000
    with pytest.raises(ValueError, match="one value per centre"):
        fiducial_mask(dataset, found, radii[:1])
    with pytest.raises(ValueError, match="not negative"):
        fiducial_mask(dataset, found, np.array([10.0, -1.0]))
    # Zero is legal: a perfectly stationary source can measure zero, and
    # masking it by exact coordinate match is not an error.
    exact = np.array([[0., 0., 0., 0.], [0., 0., 0., 1.], [10., 0., 0., 2.]])
    assert fiducial_mask(exact, [[0., 0.]], 0.0).tolist() == [True, True, False]


def test_removal_mask_tolerates_non_finite_coordinates():
    """find_fiducials drops such rows, so the mask must not choke on them."""
    locs = np.array([[0., 0., 0., 0.], [np.nan, 0., 0., 1.],
                     [10., 10., 0., 2.], [1e4, 1e4, 0., 3.]])
    mask = fiducial_mask(locs, [[0., 0.]], 50.0)
    assert mask.tolist() == [True, False, True, False]


def test_input_validation():
    with pytest.raises(ValueError):
        find_fiducials(np.zeros((10, 3)), 250.0, FIELD)
    with pytest.raises(ValueError):
        find_fiducials(np.zeros((10, 4)), -1.0, FIELD)
    with pytest.raises(ValueError, match="two finite numbers"):
        find_fiducials(np.zeros((10, 4)), 250.0, FIELD, grid_offset_nm=(np.nan, 0.0))
    with pytest.raises(ValueError, match="no finite localizations"):
        find_fiducials(np.full((10, 4), np.nan), 250.0, FIELD)
    dataset, _ = _dataset([[5_000, 5_000]])
    for bad in ({"max_expected_false_bins": 0}, {"max_expected_false_bins": -1},
                {"max_expected_false_bins": np.inf},
                {"max_locs_per_candidate": 0}, {"max_locs_per_candidate": np.inf},
                {"max_locs_per_candidate": 20.5}, {"min_reference_bins": -5},
                {"min_reference_bins": 3.7},
                {"minimum_persistence": 0.0}, {"minimum_persistence": 0.5}):
        with pytest.raises(ValueError):
            find_fiducials(dataset, 250.0, FIELD, return_report=False, **bad)


def test_field_bounds_constrain_detection_not_only_the_statistics():
    """Localizations outside the field must not become candidates."""
    rng = np.random.default_rng(5)
    n = 60_000
    background = np.column_stack([
        rng.uniform(0, 20_000, n), rng.uniform(0, 20_000, n), np.zeros(n),
        rng.integers(0, 8_000, n).astype(float)])
    outside = np.column_stack([
        rng.normal(60_000, 8.0, 20_000), rng.normal(60_000, 8.0, 20_000),
        np.zeros(20_000), np.linspace(0, 7_999, 20_000)])
    found, report = find_fiducials(np.vstack([background, outside]), MAX_DRIFT_NM,
                                   (0.0, 20_000.0, 0.0, 20_000.0))
    assert len(found) == 0
    assert report["n_locs_outside_field"] == 20_000
    assert report["n_locs_in_field"] == n


def test_a_bright_source_does_not_hide_a_faint_one():
    """The shortlist is recomputed after the background is re-estimated."""
    def cut(with_bright):
        rng = np.random.default_rng(5)
        n = 32_000
        parts = [np.column_stack([
            rng.uniform(0, 20_000, n), rng.uniform(0, 20_000, n), np.zeros(n),
            rng.integers(0, 8_000, n).astype(float)])]
        if with_bright:
            parts.append(np.column_stack([
                rng.normal(5_000, 8.0, 30_000), rng.normal(5_000, 8.0, 30_000),
                np.zeros(30_000), np.linspace(0, 7_999, 30_000)]))
        parts.append(np.column_stack([
            rng.normal(15_000, 8.0, 45), rng.normal(15_000, 8.0, 45),
            np.zeros(45), np.linspace(0, 7_999, 45)]))
        _, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD)
        faint = min(report["accepted"] + report["needs_review"] + report["rejected"],
                    key=lambda c: np.linalg.norm(c["center_nm"] - 15_000.0))
        return report, faint

    plain, plain_faint = cut(False)
    bright, bright_faint = cut(True)
    # The bright source does raise the first-pass cut ...
    assert bright["first_pass_occupancy"] > plain["first_pass_occupancy"]
    # ... but the second pass puts it back, so the faint source still proposes.
    assert bright["chance_occupancy"] == plain["chance_occupancy"]
    assert bright_faint["peak_bin_locs"] >= bright["chance_occupancy"]

def test_a_flickering_source_is_held_for_review_not_removed():
    """Spatially a fiducial, temporally two brief episodes far apart.

    No cross-time pair count can tell that from a persistent source, so it must
    not be removed automatically -- it goes to `needs_review` instead.
    """
    rng = np.random.default_rng(1)
    n_frames = 8_000
    flicker_frames = np.concatenate([np.repeat(np.arange(0, 100), 100),
                                     np.repeat(np.arange(7_900, 8_000), 100)])
    flicker = np.column_stack([
        rng.normal(5_000, 8.0, len(flicker_frames)),
        rng.normal(5_000, 8.0, len(flicker_frames)),
        np.zeros(len(flicker_frames)), flicker_frames.astype(float)])
    bead_frames = np.arange(n_frames)
    bead = np.column_stack([
        rng.normal(15_000, 8.0, n_frames), rng.normal(15_000, 8.0, n_frames),
        np.zeros(n_frames), bead_frames.astype(float)])
    background_frames = rng.integers(0, n_frames, 40_000)
    background = np.column_stack([
        rng.uniform(0, 20_000, 40_000), rng.uniform(0, 20_000, 40_000),
        np.zeros(40_000), background_frames.astype(float)])
    found, report = find_fiducials(np.vstack([flicker, bead, background]),
                                   MAX_DRIFT_NM, FIELD)
    assert len(found) == 1 and np.linalg.norm(found[0] - 15_000.0) < 30.0
    held = [item for item in report["needs_review"]
            if np.linalg.norm(item["center_nm"] - 5_000.0) < 30.0]
    assert len(held) == 1, "the flickering source should be held for review"
    assert held[0]["confidence"] == "needs_review"
    assert held[0]["reject_reason"] == "temporal_coverage"
    # It really does outscore the bead spatially -- that is the whole point.
    assert held[0]["pair_mass"] > report["accepted"][0]["pair_mass"]
    assert held[0]["temporal_coverage"] < 0.5
    assert report["accepted"][0]["temporal_coverage"] > 0.9


def test_partially_attached_beads_keep_high_confidence():
    """Brief is not flickering: a bead attached for a fifth of the movie is
    continuously present over its own span and must not be demoted."""
    rng = np.random.default_rng(9)
    n_frames = 20_000
    parts = []
    for centre, first, last in (([5_000.0, 5_000.0], 0, 4_000),
                                ([15_000.0, 15_000.0], 16_000, n_frames),
                                ([10_000.0, 15_000.0], 0, n_frames)):
        frames = np.arange(first, last)
        parts.append(np.column_stack([
            rng.normal(centre[0], 8.0, len(frames)), rng.normal(centre[1], 8.0, len(frames)),
            np.zeros(len(frames)), frames.astype(float)]))
    background_frames = rng.integers(0, n_frames, 300_000)
    parts.append(np.column_stack([
        rng.uniform(0, 20_000, 300_000), rng.uniform(0, 20_000, 300_000),
        np.zeros(300_000), background_frames.astype(float)]))
    found, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM, FIELD)
    assert len(found) == 3 and not report["needs_review"]
    assert min(item["temporal_coverage"] for item in report["accepted"]) > 0.8


def test_removal_radius_is_background_corrected():
    """The raw percentile of everything in the disc measures the disc, not the
    source: background grows as r**2 and dominates for a faint bead."""
    rng = np.random.default_rng(1)
    for bead_locs in (200, 1_000, 8_000):
        bead = np.column_stack([
            rng.normal(5_000, 10.0, bead_locs), rng.normal(5_000, 10.0, bead_locs),
            np.zeros(bead_locs), np.linspace(0, 7_999, bead_locs)])
        true_radius = np.percentile(
            np.linalg.norm(bead[:, :2] - 5_000.0, axis=1), 99)
        n_background = 60_000
        background = np.column_stack([
            rng.uniform(0, 20_000, n_background), rng.uniform(0, 20_000, n_background),
            np.zeros(n_background), rng.integers(0, 8_000, n_background).astype(float)])
        _, report = find_fiducials(np.vstack([bead, background]), MAX_DRIFT_NM, FIELD)
        candidate = min(report["accepted"] + report["needs_review"] + report["rejected"],
                        key=lambda c: np.linalg.norm(c["center_nm"] - 5_000.0))
        assert candidate["radius_p99_nm"] < 4.0 * true_radius, (
            "%d locs: true %.0f nm, reported %.0f nm"
            % (bead_locs, true_radius, candidate["radius_p99_nm"]))


def test_field_bounds_must_be_given():
    """It cannot be inferred safely, so it is not inferred silently."""
    dataset, _ = _dataset([[5_000, 5_000]])
    with pytest.raises(ValueError, match="field_bounds_nm is required"):
        find_fiducials(dataset, MAX_DRIFT_NM, None)
    guessed = field_bounds_from_data(dataset, MAX_DRIFT_NM)
    assert len(guessed) == 4 and guessed[1] > guessed[0] and guessed[3] > guessed[2]
    assert len(find_fiducials(dataset, MAX_DRIFT_NM, guessed, return_report=False)) == 1

def test_a_brighter_transient_does_not_become_high_confidence():
    """Making a transient brighter must not make it look persistent.

    A source shorter than one entropy block has no evidence of persistence at
    all, which is the conservative answer rather than a free pass -- an earlier
    version divided by a non-positive expectation there and assigned coverage
    1.0, so a 100-frame burst was automatically removed once it was bright
    enough to clear the spatial fence.
    """
    for bead_locs in (3_000, 10_000, 30_000):
        rng = np.random.default_rng(2)
        frames = np.repeat(np.arange(4_000, 4_100), bead_locs // 100)
        burst = np.column_stack([
            rng.normal(5_000, 8.0, len(frames)), rng.normal(5_000, 8.0, len(frames)),
            np.zeros(len(frames)), frames.astype(float)])
        background = np.column_stack([
            rng.uniform(0, 20_000, 60_000), rng.uniform(0, 20_000, 60_000),
            np.zeros(60_000), rng.integers(0, 8_000, 60_000).astype(float)])
        found, report = find_fiducials(np.vstack([burst, background]),
                                       MAX_DRIFT_NM, FIELD)
        assert len(found) == 0, "%d locs in 100 frames was accepted" % bead_locs
        held = min(report["needs_review"] + report["rejected"],
                   key=lambda c: np.linalg.norm(c["center_nm"] - 5_000.0))
        assert held["temporal_coverage"] < 0.1


def test_a_nearby_bead_does_not_shrink_the_removal_radius():
    """The background annulus can contain another resolved fiducial.

    Two bins away is inside it, and a plain mean density there ate the source:
    two beads 500 nm apart each reported 19 nm against a true 30 and left 17%
    of themselves behind.  The density is a median over angular sectors, so one
    contaminating direction does not move it.
    """
    def two_beads(separation_nm, bead_locs=20_000):
        rng = np.random.default_rng(4)
        parts = []
        for centre_x in (5_000.0, 5_000.0 + separation_nm):
            parts.append(np.column_stack([
                rng.normal(centre_x, 10.0, bead_locs),
                rng.normal(5_000, 10.0, bead_locs),
                np.zeros(bead_locs), np.linspace(0, 7_999, bead_locs)]))
        parts.append(np.column_stack([
            rng.uniform(0, 20_000, 60_000), rng.uniform(0, 20_000, 60_000),
            np.zeros(60_000), rng.integers(0, 8_000, 60_000).astype(float)]))
        truth = np.percentile(
            np.linalg.norm(parts[0][:, :2] - [5_000.0, 5_000.0], axis=1), 99)
        return np.vstack(parts), truth, bead_locs

    for separation in (500.0, 750.0, 2_000.0):
        dataset, true_radius, bead_locs = two_beads(separation)
        found, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD)
        assert len(found) == 2, "%.0f nm gave %d centres" % (separation, len(found))
        radii = np.asarray([item["radius_p99_nm"] for item in report["accepted"]])
        # Both directions: not inflated by background, not eaten by the neighbour.
        assert np.all(radii > 0.6 * true_radius), (
            "%.0f nm: radii %s against a true %.1f" % (separation, radii, true_radius))
        assert np.all(radii < 3.0 * true_radius)
        removed = fiducial_mask(dataset, found, radii).sum()
        assert removed > 0.95 * 2 * bead_locs, (
            "%.0f nm: only %d of %d bead localizations removed"
            % (separation, removed, 2 * bead_locs))


def test_the_fence_prevalence_assumption_is_reported_not_hidden():
    """The bulk is an upper quartile, so sources above a quarter break it.

    Contamination pushes the fence *up*: past a quarter, ``q3`` sits inside the
    sources and the fence rises above every one of them, which is a fence that
    names nothing rather than one that names everything.
    """
    for n_high, names_them in ((24, True), (25, False)):
        reference = np.concatenate([np.full(100 - n_high, 10.0),
                                    np.full(n_high, 1e5)])
        assert (outer_fence_threshold(reference) < 1e5) == names_them


def test_newly_exposed_parameters_validate_their_contracts():
    dataset, _ = _dataset([[5_000, 5_000]])
    for bad in (-1.0, 0.0, 0.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="min_effect_size"):
            find_fiducials(dataset, MAX_DRIFT_NM, FIELD, min_effect_size=bad)
        # ...and at the exported utility that actually consumes it.
        with pytest.raises(ValueError, match="min_effect_size"):
            outer_fence_threshold([1, 1, 2, 1000], min_effect_size=bad)
    for bad_drift in (0.0, -250.0, np.nan):
        with pytest.raises(ValueError, match="max_drift_nm"):
            field_bounds_from_data(dataset, bad_drift)
    with pytest.raises(ValueError, match="quantile"):
        field_bounds_from_data(dataset, MAX_DRIFT_NM, quantile=0.0)
    with pytest.raises(ValueError, match=r"shape \(N, >=2\)"):
        field_bounds_from_data(np.zeros((3, 1)), MAX_DRIFT_NM)
    # A non-numeric column past the ones that are read must not be a problem.
    mixed = np.array([(1.0, 2.0, 0.0, 3.0, "a"), (9.0, 8.0, 0.0, 4.0, "b")],
                     dtype=object)
    assert len(field_bounds_from_data(mixed, MAX_DRIFT_NM)) == 4

def test_the_removal_radius_survives_the_field_boundary():
    """Most of a corner source's measuring disc is outside the field.

    Dividing sector counts by the *geometric* area makes those sectors read as
    empty, the median density collapses to zero, and the correction vanishes.
    What the radius is *for* is removing the source, so that is what is
    asserted -- a faint corner source is measured imprecisely either way, but
    it must still come out of the data.
    """
    field = (0.0, 20_000.0, 0.0, 20_000.0)
    for centre in (10.0, 100.0, 200.0, 400.0, 10_000.0):
        rng = np.random.default_rng(7)
        bead_locs = 400
        points = np.column_stack([rng.normal(centre, 10.0, bead_locs),
                                  rng.normal(centre, 10.0, bead_locs)])
        bead = np.column_stack([points, np.zeros(bead_locs),
                                np.linspace(0, 7_999, bead_locs)])
        n_background = 60_000
        background = np.column_stack([
            rng.uniform(0, 20_000, n_background), rng.uniform(0, 20_000, n_background),
            np.zeros(n_background), rng.integers(0, 8_000, n_background).astype(float)])
        in_field = points[np.all((points >= 0) & (points < 20_000), axis=1)]
        found, report = find_fiducials(np.vstack([bead, background]),
                                       MAX_DRIFT_NM, field)
        candidate = min(report["accepted"] + report["needs_review"] + report["rejected"],
                        key=lambda c: np.linalg.norm(c["center_nm"] - centre))
        removed = (np.linalg.norm(in_field - candidate["center_nm"], axis=1)
                   <= candidate["radius_p99_nm"]).mean()
        assert removed > 0.9, (
            "centre %.0f nm from the origin removed only %.1f%% of the source"
            % (centre, 100 * removed))
        assert candidate["radius_p99_nm"] < _SOURCE_DISC * MAX_DRIFT_NM


def test_a_cloud_wider_than_max_drift_is_still_removed():
    """The measuring disc has to be wider than the thing it measures.

    A one-bin disc capped the reported radius at the bin and let the source's
    own tail into its background ring, so a bead whose cloud is wider than
    `max_drift_nm` -- the case the bin sweep says detection tolerates -- was
    silently under-removed: 83.8% at sigma 120 nm.
    """
    field = (0.0, 20_000.0, 0.0, 20_000.0)
    for sigma in (30.0, 60.0, 90.0, 120.0):
        rng = np.random.default_rng(5)
        bead_locs = 20_000
        points = np.column_stack([rng.normal(10_000, sigma, bead_locs),
                                  rng.normal(10_000, sigma, bead_locs)])
        bead = np.column_stack([points, np.zeros(bead_locs),
                                np.linspace(0, 7_999, bead_locs)])
        background = np.column_stack([
            rng.uniform(0, 20_000, 60_000), rng.uniform(0, 20_000, 60_000),
            np.zeros(60_000), rng.integers(0, 8_000, 60_000).astype(float)])
        found, report = find_fiducials(np.vstack([bead, background]),
                                       MAX_DRIFT_NM, field)
        assert len(found) == 1
        candidate = report["accepted"][0]
        removed = (np.linalg.norm(points - candidate["center_nm"], axis=1)
                   <= candidate["radius_p99_nm"]).mean()
        assert removed > 0.95, (
            "sigma %.0f nm: only %.1f%% of the source removed at radius %.1f nm"
            % (sigma, 100 * removed, candidate["radius_p99_nm"]))


def test_a_resolved_neighbour_is_neither_background_nor_cloud():
    """The measuring disc is now two bins, so a resolved neighbour can be in it.

    Its territory is masked out of both the counts and the area; without that,
    two beads 500 nm apart each reported a radius of 499 nm and removed a
    quarter of a square micrometre of specimen apiece.
    """
    field = (0.0, 20_000.0, 0.0, 20_000.0)
    for separation in (500.0, 750.0, 2_000.0):
        rng = np.random.default_rng(5)
        bead_locs = 20_000
        parts, truth = [], None
        for centre_x in (10_000.0, 10_000.0 + separation):
            points = np.column_stack([rng.normal(centre_x, 10.0, bead_locs),
                                      rng.normal(10_000, 10.0, bead_locs)])
            if truth is None:
                truth = np.percentile(
                    np.linalg.norm(points - [10_000.0, 10_000.0], axis=1), 99)
            parts.append(np.column_stack([points, np.zeros(bead_locs),
                                          np.linspace(0, 7_999, bead_locs)]))
        parts.append(np.column_stack([
            rng.uniform(0, 20_000, 60_000), rng.uniform(0, 20_000, 60_000),
            np.zeros(60_000), rng.integers(0, 8_000, 60_000).astype(float)]))
        found, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM, field)
        assert len(found) == 2, "%.0f nm gave %d centres" % (separation, len(found))
        radii = np.asarray([item["radius_p99_nm"] for item in report["accepted"]])
        assert np.all(radii > 0.6 * truth) and np.all(radii < 2.0 * truth), (
            "%.0f nm: radii %s against a true %.1f" % (separation, radii, truth))


def test_the_trial_count_does_not_move_with_the_grid_phase():
    """The Poisson model describes the field, not where the bins fall."""
    dataset, _ = _dataset([[5_000, 5_000], [12_345.6, 7_654.3], [15_000, 15_000]])
    reports = [find_fiducials(dataset, MAX_DRIFT_NM, FIELD,
                              grid_offset_nm=(offset, offset))[1]
               for offset in (0.0, 62.5, 125.0, 187.5)]
    assert len({report["n_bins"] for report in reports}) == 1
    rates = [report["background_rate_locs_per_bin"] for report in reports]
    assert max(rates) - min(rates) < 0.05 * np.mean(rates)
    assert len({report["chance_occupancy"] for report in reports}) == 1


def test_disabling_the_chance_test_disables_the_shortlist_too():
    """`None` means None: the cut it derives must go with it."""
    dataset, truth = _dataset([[5_000, 5_000], [12_345.6, 7_654.3], [15_000, 15_000]])
    _, gated = find_fiducials(dataset, MAX_DRIFT_NM, FIELD)
    found, ungated = find_fiducials(dataset, MAX_DRIFT_NM, FIELD,
                                    max_expected_false_bins=None)
    assert ungated["chance_occupancy"] < gated["chance_occupancy"]
    assert ungated["n_selected_bins"] > 10 * gated["n_selected_bins"]
    # The fence alone still gets the right answer, it just works harder.
    assert np.all(_errors(found, truth) < 25.0)


def test_the_entropy_grid_is_never_coarser_than_the_scoring_grid():
    """Once entropy gates acceptance, its resolution is policy too.

    At a fine `minimum_persistence` the stated scale is short, and a source
    that spans several of those intervals must not be refused because it
    happens to fit inside one block of the entropy grid, which holds at
    least 64 of them.
    """
    for span in (40, 80, 124):
        rng = np.random.default_rng(3)
        frames = np.repeat(np.arange(4_000, 4_000 + span), 20_000 // span)
        source = np.column_stack([
            rng.normal(5_000, 10.0, len(frames)), rng.normal(5_000, 10.0, len(frames)),
            np.zeros(len(frames)), frames.astype(float)])
        background = np.column_stack([
            rng.uniform(0, 20_000, 60_000), rng.uniform(0, 20_000, 60_000),
            np.zeros(60_000), rng.integers(0, 8_000, 60_000).astype(float)])
        found, report = find_fiducials(np.vstack([source, background]),
                                       MAX_DRIFT_NM, FIELD,
                                       minimum_persistence=1.0 / 512.0)
        assert report["n_entropy_blocks"] >= report["n_time_blocks"]
        assert len(found) == 1, "%d-frame source at 1/512 was not accepted" % span
    # A coarse scoring grid must still not drag the entropy grid down with it.
    dataset, _ = _dataset([[5_000, 5_000]])
    _, coarse = find_fiducials(dataset, MAX_DRIFT_NM, FIELD, minimum_persistence=0.25)
    assert coarse["n_entropy_blocks"] >= 64


def test_the_span_shown_to_a_reviewer_is_the_source_not_the_window():
    """Ten stray background localizations must not stretch a burst's span."""
    rng = np.random.default_rng(4)
    frames = np.repeat(np.arange(4_000, 4_100), 100)
    burst = np.column_stack([
        rng.normal(5_000, 8.0, len(frames)), rng.normal(5_000, 8.0, len(frames)),
        np.zeros(len(frames)), frames.astype(float)])
    background = np.column_stack([
        rng.uniform(0, 20_000, 60_000), rng.uniform(0, 20_000, 60_000),
        np.zeros(60_000), rng.integers(0, 8_000, 60_000).astype(float)])
    _, report = find_fiducials(np.vstack([burst, background]), MAX_DRIFT_NM, FIELD)
    candidate = min(report["accepted"] + report["needs_review"] + report["rejected"],
                    key=lambda c: np.linalg.norm(c["center_nm"] - 5_000.0))
    window_low, window_high = candidate["window_frame_span"]
    source_low, source_high = candidate["source_frame_span"]
    assert window_high - window_low > 2_000, "the window really is contaminated"
    assert 3_900 <= source_low <= 4_000 and 4_099 <= source_high <= 4_300


def test_the_report_names_why_a_fence_declined():
    """`high_shortlist_fraction` is observational: the code cannot know the
    shortlist is sources rather than dense tissue."""
    rng = np.random.default_rng(9)
    parts = []
    for index in range(400):
        cell_x, cell_y = divmod(index, 20)
        n = 20 if index < 120 else 5
        parts.append(np.column_stack([
            rng.uniform(cell_x * 250 + 20, cell_x * 250 + 230, n),
            rng.uniform(cell_y * 250 + 20, cell_y * 250 + 230, n),
            np.zeros(n), np.linspace(0, 7_999, n)]))
    found, report = find_fiducials(np.vstack(parts), MAX_DRIFT_NM,
                                   (0.0, 5_000.0, 0.0, 5_000.0))
    assert len(found) == 0
    assert report["reference_shortlist_fraction"] > 0.25
    assert report["fence_undefined_reason"] == "high_shortlist_fraction"
    # And a field with no shortlist pressure reports the other reason.
    uniform = _uniform()
    _, clean = find_fiducials(uniform, MAX_DRIFT_NM, FIELD)
    assert clean["fence_undefined_reason"] in (None, "no_separation")


# --------------------------------------------------------------------------
# review round 6: the field boundary, the grid phase, and the block axis
# --------------------------------------------------------------------------
def _bead(rng, centre, n_locs, sigma, n_frames):
    return np.column_stack([
        centre[0] + rng.normal(0, sigma, n_locs),
        centre[1] + rng.normal(0, sigma, n_locs),
        np.zeros(n_locs), rng.integers(0, n_frames, n_locs)])


def test_a_clipped_bead_with_a_neighbour_is_still_removed():
    """Boundary clipping and a resolved neighbour, together.

    The two were only ever tested apart.  Composed, a corner leaves two usable
    background sectors and the neighbour's exclusion disc eats into one of
    them, which is where the density estimate is thinnest.  Asserted on
    *removal recall*, not on a radius bound: a radius that is merely capped
    passes an upper bound while leaving part of the bead in the data.

    ``radius_p99_nm`` is a 99th percentile, so ~99% removal is the target and
    not a shortfall.  Localizations outside ``field_bounds_nm`` are counted
    separately: the method never sees them, so no radius can account for them.
    """
    field = (0.0, 20_000.0, 0.0, 20_000.0)
    for corner in (100.0, 600.0):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            background = np.column_stack([
                rng.uniform(0, 20_000, 300_000), rng.uniform(0, 20_000, 300_000),
                np.zeros(300_000), rng.integers(0, 20_000, 300_000)])
            near = _bead(rng, (corner, corner), 20_000, 10.0, 20_000)
            neighbour = _bead(rng, (corner + 500.0, corner), 20_000, 10.0, 20_000)
            dataset = np.vstack([background, near, neighbour])

            found, report = find_fiducials(dataset, MAX_DRIFT_NM, field)
            assert len(found) == 2, "corner %g seed %d: both beads resolve" % (corner, seed)
            radii = [item["radius_p99_nm"] for item in report["accepted"]]
            mask = fiducial_mask(dataset, found, radii)

            inside = ((near[:, 0] >= 0) & (near[:, 0] < 20_000)
                      & (near[:, 1] >= 0) & (near[:, 1] < 20_000))
            removed = mask[len(background):len(background) + len(near)][inside].mean()
            assert removed > 0.97, (
                "corner %g seed %d: only %.1f%% of the clipped bead removed, radii %s"
                % (corner, seed, 100 * removed, radii))
            assert (~mask[:len(background)]).mean() > 0.99, "background is not swept up"


def test_the_thinness_of_a_background_estimate_is_reported():
    """A corner leaves fewer usable sectors, and the report says how many."""
    field = (0.0, 20_000.0, 0.0, 20_000.0)
    rng = np.random.default_rng(5)
    background = np.column_stack([
        rng.uniform(0, 20_000, 300_000), rng.uniform(0, 20_000, 300_000),
        np.zeros(300_000), rng.integers(0, 20_000, 300_000)])
    dataset = np.vstack([background,
                         _bead(rng, (100.0, 100.0), 20_000, 10.0, 20_000),
                         _bead(rng, (10_000.0, 10_000.0), 20_000, 10.0, 20_000)])
    _, report = find_fiducials(dataset, MAX_DRIFT_NM, field)
    by_x = sorted(report["accepted"], key=lambda item: item["center_nm"][0])
    assert by_x[0]["n_background_sectors"] < by_x[-1]["n_background_sectors"]
    assert by_x[-1]["n_background_sectors"] == 8, "an interior bead sees the whole ring"


def test_a_bead_on_a_grid_crossing_is_found_like_any_other():
    """The fence is a threshold, not a verdict.

    A source centred on a grid crossing is divided among four bins and clears
    the fence in none of them; deciding "nothing separates" against the raw
    reference threw it away, while the same bead half a bin over was named.
    """
    field = (0.0, 10_000.0, 0.0, 10_000.0)
    verdicts = {}
    for label, centre in (("crossing", (5_000.0, 5_000.0)),
                          ("cell centre", (5_125.0, 5_125.0))):
        rng = np.random.default_rng(7)
        background = np.column_stack([
            rng.uniform(0, 10_000, 64_000), rng.uniform(0, 10_000, 64_000),
            np.zeros(64_000), rng.integers(0, 20_000, 64_000)])
        dataset = np.vstack([background, _bead(rng, centre, 300, 12.0, 20_000)])
        found, report = find_fiducials(dataset, MAX_DRIFT_NM, field)
        assert np.isfinite(report["score_threshold"]), label
        verdicts[label] = (len(found), float(np.min(np.linalg.norm(
            found - np.asarray(centre), axis=1))) if len(found) else np.inf)
    assert verdicts["crossing"][0] == verdicts["cell centre"][0] == 1, verdicts
    assert verdicts["crossing"][1] < 25.0 and verdicts["cell centre"][1] < 25.0


def _dense_region_field():
    """A bead-free field with one contiguous dense region in the middle."""
    rng = np.random.default_rng(9)
    parts = []
    for cell_x in range(40):
        for cell_y in range(40):
            dense = 16 <= cell_x < 24 and 16 <= cell_y < 24
            n_locs = 20 if dense else (5 if (cell_x + cell_y) % 2 else 4)
            parts.append(np.column_stack([
                rng.uniform(cell_x * 250 + 20, (cell_x + 1) * 250 - 20, n_locs),
                rng.uniform(cell_y * 250 + 20, (cell_y + 1) * 250 - 20, n_locs),
                np.zeros(n_locs), np.linspace(0, 7_999, n_locs)]))
    return np.vstack(parts)


@pytest.mark.parametrize("merge_edges", [True, False])
def test_dense_tissue_cannot_answer_for_the_field_on_its_placement(merge_edges):
    """The compact rule is what keeps the crossing fix from naming tissue.

    A candidate may declare that the field separates only if its own
    neighbourhood holds a 2x2 block of shortlisted bins or less -- all a source
    one bin wide can cover at any phase.  A dense region is free to place its
    window wherever the field is thickest, and must not clear a fence built
    from bins that were not.

    Both merge modes, because compactness must be a property of the source and
    not of the caller's setting: read off the merged component instead, it was
    1 for every candidate with merge_edges off and the whole region qualified.
    """
    found, report = find_fiducials(_dense_region_field(), MAX_DRIFT_NM,
                                   (0.0, 10_000.0, 0.0, 10_000.0),
                                   merge_edges=merge_edges)
    assert len(found) == 0
    assert report["fence_undefined_reason"] == "no_separation"
    # Inside the region -- not on its rim, where a 2x2 shortlisted corner is
    # genuinely all there is -- every neighbour is shortlisted too, so no
    # candidate there can answer for the field.
    interior = [item for item in report["rejected"]
                if 4_500 < item["center_nm"][0] < 5_500
                and 4_500 < item["center_nm"][1] < 5_500]
    assert interior, "the dense region does propose candidates"
    assert all(item["footprint_bins"] == 9 for item in interior), (
        "tissue fills its neighbourhood: %s"
        % sorted({item["footprint_bins"] for item in interior}))


def test_the_footprint_of_a_source_does_not_depend_on_merge_edges():
    """A bead occupies the same bins however the caller groups them."""
    dataset, _truth = _dataset([[5_000, 5_000]])
    footprints = {}
    for merge_edges in (True, False):
        _, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD,
                                   merge_edges=merge_edges)
        item = report["accepted"][0]
        footprints[merge_edges] = item["footprint_bins"]
        assert item["footprint_bins"] <= 4, "one bead is compact either way"
    assert footprints[True] == footprints[False]


def test_a_fractional_field_still_names_a_bead_in_the_middle_of_it():
    """Every edge cell of a 4.1 x 4.1-bin field is a sliver.

    Counted as whole bins they are a mode of their own: the log-space IQR
    spanned both modes and the fence rose into the billions, which rejected a
    1,000-localization bead sitting in the middle of an otherwise uniform
    field.  Exposure is what makes the two comparable.
    """
    side = 4.1 * MAX_DRIFT_NM
    rng = np.random.default_rng(4)
    n_background = int(round(47 * 4.1 * 4.1))
    background = np.column_stack([
        rng.uniform(0, side, n_background), rng.uniform(0, side, n_background),
        np.zeros(n_background), rng.integers(0, 20_000, n_background)])
    dataset = np.vstack([background,
                         _bead(rng, (side / 2, side / 2), 1_000, 12.0, 20_000)])
    found, report = find_fiducials(dataset, MAX_DRIFT_NM, (0.0, side, 0.0, side))
    assert len(found) == 1, report["fence_undefined_reason"]
    assert np.linalg.norm(found[0] - side / 2) < 25.0
    # The rate is per whole bin, and the slivers neither inflate nor deflate it.
    assert 42.0 < report["background_rate_locs_per_bin"] < 56.0
    # And the reference is built from the bins that are whole.
    assert report["n_reference_bins"] == 16


def test_a_sliver_of_a_bin_is_not_tested_as_a_whole_one():
    """A cell one tenth inside the field expects one tenth of the rate."""
    from comet_fiducials.histogram import _occupancy_by_exposure
    exposure = np.array([1.0, 0.5, 0.1])
    cuts = _occupancy_by_exposure(40.0, exposure, 1_000, 1.0)
    assert cuts[0] > cuts[1] > cuts[2] >= 2
    assert np.all(cuts >= 2), "a bin still needs two localizations to carry a pair"


def test_entropy_blocks_do_not_scale_with_the_frame_range():
    """The same sparse treatment the pair mass already had.

    ``minimum_persistence`` is a fraction, so a long acquisition makes the
    entropy grid enormous; the dense form allocated one entry per block per
    candidate and a five-million-frame acquisition cost 87 MB for one bead.
    """
    import tracemalloc
    rng = np.random.default_rng(3)
    n_frames = 5_000_001
    background = np.column_stack([
        rng.uniform(0, 20_000, 20_000), rng.uniform(0, 20_000, 20_000),
        np.zeros(20_000), rng.integers(0, n_frames, 20_000)])
    dataset = np.vstack([background,
                         _bead(rng, (5_000.0, 5_000.0), 20_000, 12.0, n_frames)])

    def peak_bytes(minimum_persistence):
        tracemalloc.start()
        try:
            _, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD,
                                       minimum_persistence=minimum_persistence)
            return tracemalloc.get_traced_memory()[1], report
        finally:
            tracemalloc.stop()

    coarse, coarse_report = peak_bytes(1.0 / 64.0)
    fine, fine_report = peak_bytes(1e-7)
    assert fine_report["n_entropy_blocks"] > 1_000 * coarse_report["n_entropy_blocks"]
    dense_growth = fine_report["n_entropy_blocks"] * 8
    assert dense_growth > 1e7, "the dense form would have grown by tens of MB per candidate"
    assert fine < 2.0 * coarse, (
        "peak grew from %.0f MB to %.0f MB with the entropy block count"
        % (coarse / 1e6, fine / 1e6))


def test_the_source_frame_span_stays_inside_the_acquisition():
    """It is a block-resolution estimate, and the last block overhangs the end."""
    rng = np.random.default_rng(6)
    n_frames = 20_000
    background = np.column_stack([
        rng.uniform(0, 20_000, 100_000), rng.uniform(0, 20_000, 100_000),
        np.zeros(100_000), rng.integers(0, n_frames, 100_000)])
    dataset = np.vstack([background,
                         _bead(rng, (5_000.0, 5_000.0), 20_000, 12.0, n_frames)])
    _, report = find_fiducials(dataset, MAX_DRIFT_NM, FIELD)
    first, last = int(dataset[:, 3].min()), int(dataset[:, 3].max())
    assert report["accepted"], "the bead is found"
    for item in report["accepted"] + report["rejected"] + report["needs_review"]:
        low, high = item["source_frame_span"]
        assert first <= low <= high <= last, item["source_frame_span"]
