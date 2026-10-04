"""The progress callback, cancellation and run details of comet_run_kd.

An application running COMET for minutes needs to show where the run is, to
stop it, and to know afterwards what the run did -- without COMET printing,
plotting or touching the caller's data on the way.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import comet
from comet.core.drift_optimizer import RunDetails, comet_run_kd
from comet.tests.conftest import drift_residual_nm, make_drifting_dataset
from comet.tests.test_drift_correctness import RUN_KWARGS, TOLERANCE_NM


class Stop(Exception):
    """Raised by a callback to cancel a run."""


def _recorder():
    calls = []

    def progress(stage, info):
        calls.append((stage, dict(info)))

    return calls, progress


def _stages(calls):
    return [stage for stage, _ in calls]


class TestStages:
    def test_stages_arrive_in_pipeline_order(self, drifting_dataset):
        locs, _ = drifting_dataset
        calls, progress = _recorder()

        comet_run_kd(dataset=locs.copy(), progress=progress, **RUN_KWARGS)

        stages = _stages(calls)
        first = stages.index
        assert stages[:3] == ["segmentation", "pairs_start", "pairs_done"]
        assert first("pairs_done") < first("run_start") < first("evaluation") < first("run_end")
        assert stages[-3:] == ["interpolation", "apply", "done"]
        assert set(stages) == {"segmentation", "pairs_start", "pairs_done", "run_start",
                               "evaluation", "run_end", "interpolation", "apply", "done"}

    def test_every_run_is_bracketed(self, drifting_dataset):
        locs, _ = drifting_dataset
        calls, progress = _recorder()

        comet_run_kd(dataset=locs.copy(), progress=progress, **RUN_KWARGS)

        runs = [(stage, info["run"]) for stage, info in calls if stage in ("run_start", "run_end")]
        assert len(runs) % 2 == 0
        for (start, a), (end, b) in zip(runs[0::2], runs[1::2]):
            assert (start, end) == ("run_start", "run_end")
            assert a == b

    def test_evaluations_count_up_and_sigma_shrinks(self, drifting_dataset):
        locs, _ = drifting_dataset
        calls, progress = _recorder()

        comet_run_kd(dataset=locs.copy(), progress=progress, **RUN_KWARGS)

        evaluations = [info for stage, info in calls if stage == "evaluation"]
        counts = [info["n_evaluations"] for info in evaluations]
        assert counts == list(range(1, len(counts) + 1))
        sigmas = [info["sigma_nm"] for stage, info in calls if stage == "run_start"]
        assert sigmas[0] == pytest.approx(RUN_KWARGS["initial_sigma_nm"])
        assert all(later < earlier for earlier, later in zip(sigmas, sigmas[1:]))
        assert all(np.isfinite(info["cost"]) for info in evaluations)

    def test_pairs_done_reports_the_pairs_the_optimizer_used(self, drifting_dataset):
        locs, _ = drifting_dataset
        calls, progress = _recorder()

        comet_run_kd(dataset=locs.copy(), progress=progress, **RUN_KWARGS)

        reported = [info for stage, info in calls if stage == "pairs_done"][0]["n_pairs"]
        expected = len(cKDTree(locs[:, :3]).query_pairs(RUN_KWARGS["max_drift_nm"]))
        assert reported == expected

    def test_done_carries_the_details(self, drifting_dataset):
        locs, _ = drifting_dataset
        calls, progress = _recorder()

        comet_run_kd(dataset=locs.copy(), progress=progress, **RUN_KWARGS)

        assert isinstance(calls[-1][1]["details"], RunDetails)

    def test_a_progress_callback_does_not_make_the_run_print(self, drifting_dataset, capsys):
        locs, _ = drifting_dataset
        comet_run_kd(dataset=locs.copy(), progress=lambda stage, info: None, **RUN_KWARGS)
        assert capsys.readouterr().out == ""


class TestCancellation:
    @pytest.mark.parametrize("stage", ["segmentation", "pairs_start", "pairs_done", "run_start",
                                       "evaluation", "run_end", "interpolation"])
    def test_raising_stops_the_run_and_leaves_the_data_alone(self, drifting_dataset, stage):
        locs, _ = drifting_dataset
        dataset = locs.copy()

        def progress(reached, info):
            if reached == stage:
                raise Stop

        with pytest.raises(Stop):
            comet_run_kd(dataset=dataset, progress=progress, **RUN_KWARGS)
        np.testing.assert_array_equal(dataset, locs)

    def test_cancel_takes_effect_at_the_next_evaluation(self, drifting_dataset):
        locs, _ = drifting_dataset
        seen = []

        def progress(stage, info):
            if stage == "evaluation":
                seen.append(info["n_evaluations"])
                if info["n_evaluations"] == 3:
                    raise Stop

        with pytest.raises(Stop):
            comet_run_kd(dataset=locs.copy(), progress=progress, **RUN_KWARGS)
        assert seen == [1, 2, 3]

    def test_the_apply_stage_comes_before_the_data_changes(self, drifting_dataset):
        locs, _ = drifting_dataset
        dataset = locs.copy()
        at_apply = []

        def progress(stage, info):
            if stage == "apply":
                at_apply.append(dataset.copy())

        comet_run_kd(dataset=dataset, progress=progress, **RUN_KWARGS)

        np.testing.assert_array_equal(at_apply[0], locs)
        assert not np.array_equal(dataset, locs)


class TestDetails:
    def test_returned_last_in_every_combination(self, drifting_dataset):
        locs, _ = drifting_dataset

        drift, details = comet_run_kd(dataset=locs.copy(), return_details=True, **RUN_KWARGS)
        assert drift.ndim == 2 and isinstance(details, RunDetails)

        drift, corrected, details = comet_run_kd(dataset=locs.copy(), return_corrected_locs=True,
                                                 return_details=True, **RUN_KWARGS)
        assert corrected.shape == locs.shape and isinstance(details, RunDetails)

    def test_without_the_flag_the_return_is_unchanged(self, drifting_dataset):
        locs, _ = drifting_dataset
        assert isinstance(comet_run_kd(dataset=locs.copy(), **RUN_KWARGS), np.ndarray)

    def test_details_describe_the_run(self, drifting_dataset):
        locs, gt_drift = drifting_dataset
        calls, progress = _recorder()

        drift, details = comet_run_kd(dataset=locs.copy(), return_details=True,
                                      progress=progress, **RUN_KWARGS)

        n_segments = details.segmentation.n_segments
        assert details.backend == "cpu"
        assert details.knot_frames.shape == (n_segments,)
        assert details.knot_drift_nm.shape == (n_segments, 3)
        assert details.n_pairs == len(cKDTree(locs[:, :3]).query_pairs(RUN_KWARGS["max_drift_nm"]))
        assert details.n_runs == _stages(calls).count("run_start")
        assert details.n_evaluations == _stages(calls).count("evaluation")
        assert details.n_failures == 0 and not details.auto_downsampled
        assert details.sigma_initial_nm == RUN_KWARGS["initial_sigma_nm"]
        assert details.sigma_target_nm == RUN_KWARGS["target_sigma_nm"]
        assert set(details.timings_s) >= {"segmentation", "pairs", "optimization", "interpolation"}

    def test_accepted_sigma_is_within_the_documented_range(self, drifting_dataset):
        locs, _ = drifting_dataset
        _, details = comet_run_kd(dataset=locs.copy(), return_details=True, **RUN_KWARGS)

        target = RUN_KWARGS["target_sigma_nm"]
        assert 1.0 <= details.sigma_accepted_nm <= 1.5 * target + 1e-9
        assert details.sigma_last_nm < details.sigma_accepted_nm

    def test_the_knots_determine_the_returned_drift(self, drifting_dataset):
        """A caller holding only the knots can rebuild the per-frame drift."""
        from comet.core.interpolation import interpolate_drift

        locs, _ = drifting_dataset
        drift, details = comet_run_kd(dataset=locs.copy(), return_details=True, **RUN_KWARGS)

        valid = np.isfinite(details.knot_drift_nm[:, 0])
        rebuilt = interpolate_drift(details.knot_frames[valid], details.knot_drift_nm[valid],
                                    drift[:, 3].astype(int), method=RUN_KWARGS["interpolation_method"])
        np.testing.assert_allclose(drift[:, :3], rebuilt, atol=1e-12)

    def test_details_object_is_exported(self):
        assert "RunDetails" in comet.__all__
        assert comet.RunDetails is RunDetails


class TestRandomState:
    CAPPED = dict(RUN_KWARGS, segmentation_mode=1, segmentation_var=60, max_locs_per_segment=40)

    def test_a_seed_makes_a_capped_run_reproducible(self, drifting_dataset):
        locs, _ = drifting_dataset
        a = comet_run_kd(dataset=locs.copy(), random_state=7, **self.CAPPED)
        b = comet_run_kd(dataset=locs.copy(), random_state=7, **self.CAPPED)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_choose_different_localizations(self, drifting_dataset):
        locs, _ = drifting_dataset
        _, a = comet_run_kd(dataset=locs.copy(), random_state=1, return_details=True, **self.CAPPED)
        _, b = comet_run_kd(dataset=locs.copy(), random_state=2, return_details=True, **self.CAPPED)
        assert not np.array_equal(a.segmentation.loc_valid, b.segmentation.loc_valid)

    def test_a_generator_is_accepted(self, drifting_dataset):
        locs, gt_drift = drifting_dataset
        drift = comet_run_kd(dataset=locs.copy(), random_state=np.random.default_rng(3), **self.CAPPED)
        assert drift_residual_nm(drift, gt_drift) < TOLERANCE_NM

    def test_no_seed_still_uses_the_global_random_state(self, drifting_dataset):
        locs, _ = drifting_dataset
        np.random.seed(11)
        _, a = comet_run_kd(dataset=locs.copy(), return_details=True, **self.CAPPED)
        np.random.seed(11)
        _, b = comet_run_kd(dataset=locs.copy(), return_details=True, **self.CAPPED)
        np.testing.assert_array_equal(a.segmentation.loc_valid, b.segmentation.loc_valid)


def test_cancel_works_on_every_backend(drifting_dataset, any_backend):
    """The callback sits around the cost function, so no backend can ignore it."""
    locs, _ = drifting_dataset
    dataset = locs.copy()

    def progress(stage, info):
        if stage == "evaluation" and info["n_evaluations"] == 2:
            raise Stop

    with pytest.raises(Stop):
        comet_run_kd(dataset=dataset, progress=progress, **dict(RUN_KWARGS, mode=any_backend))
    np.testing.assert_array_equal(dataset, locs)
