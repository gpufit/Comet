"""Mode 0/1 segmentation against the 1.1.0 implementation.

1.2 rewrote the localizations-per-window segmenter, which was quadratic in the
number of windows, as a single pass. The 1.1.0 code is kept here verbatim as the
reference: the rewrite must reproduce it exactly, random choices included.
"""

import numpy as np
import pytest
from typing import Optional, Dict  # noqa: F401 -- the reference's annotations

from comet.core.segmenter import (SegmentationResult, _group_by_frame,  # noqa: F401
                                  segment_by_frame_windows, segment_by_num_locs_per_window)


# ---- 1.1.0, verbatim -------------------------------------------------------
def reference_segment_by_num_locs_per_window(loc_frames: np.ndarray, min_n_locs_per_window: int,
                                   max_locs_per_segment = None,
                                   return_param_dict: bool = False) -> SegmentationResult:
    """
    Segments by collecting a minimum number of localizations per window.
    Once the threshold is met and enough locs remain, a new segment is created.
    This method ensures that each segment has at least `min_n_locs_per_window` localizations,
    while also trying to avoid creating segments that are too small at the end of the dataset.
    If `max_locs_per_segment` is set, a random subset of that size is chosen from each segment.
    This method is particularly useful for datasets with varying localization densities over time.
    Parameters:
    loc_frames (np.ndarray): Array of frame numbers for each localization.
    min_n_locs_per_window (int): Minimum number of localizations per segment.
    max_locs_per_segment (Optional[int]): Maximum number of localizations per segment. If None, all locs are used.
    return_param_dict (bool): Whether to return a dictionary of segmentation parameters.
    Returns:
    SegmentationResult: A dataclass containing segmentation results and parameters.
    """
    loc_frames = np.asarray(loc_frames, dtype=int)
    n_locs = len(loc_frames)

    if max_locs_per_segment is not None and max_locs_per_segment < 1:  # downsampling in percentage
        max_locs_per_segment = int(min_n_locs_per_window * max_locs_per_segment)

    unique_frames, frame_to_indices = _group_by_frame(loc_frames)
    loc_segments = np.full(n_locs, -1, dtype=int)  # Default to -1 for safety
    segment_counter = 0
    n_locs_in_current_segment = 0
    current_segment_indices = []
    start_frames, end_frames, locs_per_segment = [], [], []

    for i, frame in enumerate(unique_frames):
        indices = frame_to_indices[frame]
        n_locs_this_frame = len(indices)
        remaining_locs = n_locs - (len(current_segment_indices) + n_locs_this_frame + np.sum(locs_per_segment))

        # Add frame to current segment if:
        # - It fills the current segment to threshold
        # - AND there are enough locs left for another segment (or it's the last frame)
        if (n_locs_in_current_segment + n_locs_this_frame >= min_n_locs_per_window) and \
                (remaining_locs >= min_n_locs_per_window or i == len(unique_frames) - 1):
            current_segment_indices.extend(indices)
            n_locs_in_current_segment += n_locs_this_frame

            loc_segments[current_segment_indices] = segment_counter
            start_frames.append(loc_frames[current_segment_indices[0]])
            end_frames.append(loc_frames[current_segment_indices[-1]])
            locs_per_segment.append(len(current_segment_indices))

            segment_counter += 1
            current_segment_indices = []
            n_locs_in_current_segment = 0
        else:
            # Defer frame to current segment
            current_segment_indices.extend(indices)
            n_locs_in_current_segment += n_locs_this_frame

    n_segments = segment_counter
    center_frames = np.zeros(n_segments)
    loc_valid = np.zeros(n_locs, dtype=bool)

    for i in range(n_segments):
        segment_indices = np.where(loc_segments == i)[0]
        if max_locs_per_segment and len(segment_indices) > max_locs_per_segment:
            selected = np.random.choice(segment_indices, max_locs_per_segment, replace=False)
        else:
            selected = segment_indices
        loc_valid[selected] = True
        locs_per_segment[i] = len(selected)
        center_frames[i] = np.mean(loc_frames[selected])

    out_dict = None
    if return_param_dict:
        n_locs_valid = loc_valid.sum()
        out_dict = {
            "n_segments": n_segments,
            "min_n_locs_per_window": min_n_locs_per_window,
            "frames_per_window": -1,
            "start_frames": np.array(start_frames),
            "end_frames": np.array(end_frames),
            "locs_per_segment": np.array(locs_per_segment),
            "n_locs": n_locs,
            "n_locs_valid": n_locs_valid,
            "n_locs_invalid": n_locs - n_locs_valid,
            "center_frames": center_frames
        }

    return SegmentationResult(loc_segments, loc_valid, center_frames, n_segments, out_dict)

# ---------------------------------------------------------------------------


def _frames(seed, n_locs, n_frames, gaps=False):
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, n_frames, n_locs)
    if gaps:
        frames = frames[(frames // 7) % 3 != 1]     # whole blocks of empty frames
    return frames


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("min_locs", [1, 13, 200, 5000])
@pytest.mark.parametrize("cap", [None, 10, 0.5])
@pytest.mark.parametrize("gaps", [False, True])
def test_matches_the_1_1_segmenter_exactly(seed, min_locs, cap, gaps):
    frames = _frames(seed, 6000, 900, gaps=gaps)

    np.random.seed(seed)
    expected = reference_segment_by_num_locs_per_window(frames, min_locs, cap, return_param_dict=True)
    np.random.seed(seed)
    actual = segment_by_num_locs_per_window(frames, min_locs, cap, return_param_dict=True)

    assert actual.n_segments == expected.n_segments
    np.testing.assert_array_equal(actual.loc_segments, expected.loc_segments)
    np.testing.assert_array_equal(actual.loc_valid, expected.loc_valid)
    np.testing.assert_array_equal(actual.center_frames, expected.center_frames)
    for key, value in expected.out_dict.items():
        np.testing.assert_array_equal(actual.out_dict[key], value, err_msg=key)


def test_a_fractional_cap_works_in_frame_windows():
    """mode 2 turned a fraction into a float sample size, which numpy refuses."""
    frames = _frames(0, 5000, 400)
    result = segment_by_frame_windows(frames, 20, 0.5, return_param_dict=True)
    assert result.loc_valid.sum() < len(frames)
    assert np.issubdtype(result.out_dict["locs_per_segment"].dtype, np.integer)
