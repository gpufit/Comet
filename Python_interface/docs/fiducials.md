# Automatic fiducial detection

`comet_fiducials` finds fiducial beads in a localization dataset from **one 2D
histogram**, binned at the maximum expected drift — the same `max_drift_nm`
COMET takes as its pair-search radius.

It is a **standalone package**. It imports nothing but NumPy and SciPy, in
particular nothing from `comet`; COMET never calls it, and nothing in the
drift-correction path changes if you ignore it. `comet_fiducials/histogram.py`
has no intra-package imports either, so one file is enough to vendor it
elsewhere.

```python
from comet_fiducials import find_fiducials, fiducial_mask

centers_nm, report = find_fiducials(locs, max_drift_nm=250,
                                    field_bounds_nm=camera_fov_nm)

# remove them, each at its own measured cloud radius
radii = [c["radius_p99_nm"] for c in report["accepted"]]
clean = locs[~fiducial_mask(locs, centers_nm, radii)]
```

`locs` is the usual `(N, >=4)` array of `[x_nm, y_nm, z_nm, frame, ...]`; only
x, y and frame are read. `centers_nm` holds the **high-confidence** centres,
strongest first. `report` is a dict — see
[What the report contains](#what-the-report-contains).

!!! warning "The field of view is required, and deliberately not guessed"
    The Poisson trial count and the background rate both describe a field of
    view, and it cannot be inferred safely from localizations alone: an empty
    corner of the camera and an invalid coordinate look identical, and the two
    call for opposite treatment. Measured — with a persistent 400-localization
    fiducial in an empty part of the camera, a box inferred from the
    localizations discarded all 400 of them and found nothing, while the real
    camera bounds recovered it.

    Pass the camera field of view. If you genuinely have no metadata,
    `field_bounds_from_data(locs, max_drift_nm)` returns a guess, so that
    falling back to one is a visible line of code rather than a default.

## Why you would want this

Fiducial beads are static point sources: they are the densest, longest-lived
clusters in an uncorrected field, and they dominate the cost of any pair-based
drift estimator. On one measured real crop, **seventeen beads carried 99.7% of
the whole field's cross-time localization-pair mass** (99.8% counting in 3D).
Finding them automatically is worth doing whether you want to use them as a
drift reference or remove them so the biological structure drives the
correction.

## How it works

### 1. One histogram, binned at the drift scale

The bin size is a derivation, not a tuning knob. A bead is a *static* point
source, so the only thing that spreads its localization cloud in uncorrected
data is the stage drift — the cloud diameter is bounded by `max_drift_nm`.
Binning at exactly that scale is the unique choice that puts a whole bead
inside one cell: finer bins shatter the cloud across cells, coarser bins dilute
it into the surrounding structure.

Only occupied cells are ever materialised, so the cost does not depend on the
coordinate bounding box. A single stray coordinate cannot make the histogram
allocate gigabytes.

### 2. The bins chance would not have produced

Given the field's own localization density there is an occupancy above which a
uniform field would not be expected to reach *anywhere* — the same
`expected_false_bins` criterion used at selection. Every bin above it is a
proposal.

This is derived from the data, not a fixed percentile, and that matters for
more than tidiness: a percentile shortlist can only ever propose that fraction
of the field, so twenty beads among a thousand occupied bins could not all be
proposed by a p99 cut. There is no cap on how many beads the method can find.

### 3. Merge on the edges, seed at local maxima, then leave the grid

A bead sitting on a grid line lights up two or four neighbouring bins, so
touching proposals are merged (8-connectivity). Each **local-maximum bin**
inside the merged blob then seeds its own candidate — two beads a couple of
bins apart light up one connected blob, and a single window centred on its
joint centre of mass would sit between them and catch neither.

Each seed's centre is then walked out of the grid by a **box mean shift**: a
bin-sized window repeatedly recentred on its own centre of mass. Across a full
bin of grid phase the centres move by only a few nanometres and the accepted
set does not change at all; without the merge and the mean shift, the same
sweep produces between zero and three spurious detections and three times the
centre error.

### 4. Score by cross-time pair mass

A bead does not merely hold many localizations; it holds many localizations
**spread over the acquisition**, and only pairs whose members fall in different
time blocks carry drift information at all. Candidates are scored by that
cross-time count and the outliers are kept.

!!! note "What `pair_mass` is, exactly"
    `pair_mass` counts cross-time pairs inside the candidate's bin-sized
    **square** window, treating the window as a complete graph. It is an
    **upper bound on, and a proxy for**, a radius-limited count: the square's
    diagonal is `sqrt(2) x max_drift_nm`, so two members can be 41% further
    apart than the radius, and only XY is used. That is deliberate — the proxy
    has the closed form `(n² − Σ n_b²) / 2`, so it costs one `bincount` instead
    of a KD-tree, and for ranking a bead against ordinary field it behaves like
    the real count. Where an exact radius-limited number is wanted,
    `pair_mass_fine` counts real neighbours at `pair_radius_nm` with a
    KD-tree.

**Two half-offset time grids.** Splitting the acquisition into blocks makes the
cross-time count depend on where the boundaries fall: a short burst sitting
inside one block scores nothing, while the same burst shifted to straddle a
boundary scores as though it were persistent — enough, in an 8,000-frame
acquisition, for a 200-frame burst to outrank a real persistent bead. Every
cross-time count is therefore evaluated on **two grids offset by half a block,
and the smaller is kept**. A source narrower than half a block falls inside a
block on one of the two grids; a persistent source scores the same on both.

!!! warning "This measures temporal spread, not continuous presence"
    Half a block is `minimum_persistence` of the acquisition, and that is a
    **stated policy, not a derivation** — it is the answer to "how much of the
    movie must a source span before it counts as persistent?", and it is the
    only place a timescale enters the method. A source spanning less than that
    *contiguously* scores near zero. A source that appears in two brief
    episodes far apart does not: its localizations really are in different time
    blocks, and no cross-time statistic can tell that from a bead. Such a
    source is classed **`needs_review`** on its `temporal_coverage` and is not
    removed automatically. That measure is the normalised entropy of its
    counts over a grid that is **at least as fine as the scoring grid and never
    coarser than 64 blocks** — with the expected background subtracted per
    block first — divided by the entropy a *contiguous* source of the same span
    would show. The floor is there because at a coarse scoring grid every
    source looks uniform; the "never coarser" half is there because once
    entropy gates acceptance, a source shorter than one entropy block would be
    refused however short `minimum_persistence` says a persistent source may
    be.

    `minimum_persistence` sets the block width directly rather than being
    rounded into an integer block count, and the value actually achieved after
    rounding to whole frames comes back as
    `report["effective_minimum_persistence"]`.

By default the KD-tree diagnostics (`pair_mass_fine`, `concentration`) are
skipped, because nothing in the selection reads them and they are the bulk of
the runtime. Pass `compute_fine_pair_mass=True` to get them; they are `nan`
otherwise.

Candidates too large to count pairs on directly are subsampled **uniformly in
time** and rescaled by `stride²`. Striding in time preserves the temporal
spread, and hence the cross-time fraction; a spatial or head-of-list subsample
would not. Measured accuracy of that estimator: 0.006% median error against the
exact closed form.

## Selection

Three conditions, all derived rather than tuned.

**A log-space outer fence** on the pair mass, as one continuous rule:

* Tukey's outer fence, `expm1(q3 + 3·IQR)` of `log1p` — the shape of the bulk
  decides how far out to go;
* **floored at `min_effect_size` × the bulk** (default 25) — an outlier has to
  be *large*, not merely unusual. Since pair mass scales as occupancy squared,
  25× in pair mass is five times the localizations of ordinary field.

The fence is a threshold and not a verdict: it is `inf` only when there are
fewer than two positive bins to place one against. Whether anything separates
is decided against the *candidates*, because the reference is fixed grid bins
and a fiducial that lands on a grid crossing is divided among four of them and
clears the fence in none — deciding it against the reference threw exactly that
fiducial away. A candidate may answer for the field only if it is **compact**
— its own 3×3 neighbourhood holding a 2×2 block of shortlisted bins or less,
which is all a source one bin wide can cover at any grid phase. Dense tissue
fills all nine and cannot answer on the strength of where its window happened
to land. Measured on the neighbourhood and not on the merged component, so it
says the same thing whichever way `merge_edges` is set.

The floor is what makes this safe, and it is why there is no special case for a
degenerate reference. A reference whose middle half has a *small but nonzero*
spread — a sparse field, where bin pair masses take only a handful of distinct
values — puts Tukey's fence a few percent above the bulk, and a broad dense
structural region is then named wholesale: measured, a bead-free field with a
contiguous dense region produced **34 fiducials**. A reference with *no* spread
puts the fence exactly at the bulk. Both are the same failure at different
scales, and an earlier version that branched on `q1 == q3` exactly was
discontinuous across it — adding `1e-12` of jitter to the bulk flipped it from
naming nothing to naming an entire second mode.

Nothing here caps how many fiducials may be named: two equally strong sources
in a flat field are both found, and so are twelve.

The floor is the most consequential number in the method, so it is measured
rather than assumed and it is exposed as `min_effect_size`. Against a bead-free
field with a contiguous dense region at 19× the bulk pair mass, and a bead against
a 47-localizations-per-bin background:

| floor | false positives | faintest bead found |
| --- | --- | --- |
| 2 | 34 | 78 locs |
| 5 | 34 | 78 |
| 10 | 34 | 312 |
| **25** | **0** | **312** |
| 100 | 0 | 1250 |

25 is the smallest floor that removes the false positives; going higher buys no
measured safety and costs four times the sensitivity. The margin over the
hardest negative tested is only 1.3×, which is the sharpest reason more
structured bead-free datasets are needed before this is called production-ready.

!!! note "The fence assumes fiducials are a rare tail — under a quarter of bins"
    The bulk is an upper quartile, so once sources occupy more than 25% of the
    positive reference, `q3` sits inside the source population and the
    threshold rises above every source: 24 high scores in 100 gives a usable
    threshold, 25 puts the threshold above the sources themselves.
    Contamination pushes `q3` *up*, so the failure is a fence that is too
    strict rather than too permissive. The report says so rather than failing
    silently — `reference_shortlist_fraction`, and `fence_undefined_reason` of
    `high_shortlist_fraction` when nothing is named — named for what is
    observed, since the code cannot know whether that shortlist is sources or
    dense tissue.

    The shortlist fraction is not the *source* fraction: on the real crop it is
    0.34, because most of the shortlist there is ordinary dense tissue rather
    than beads, and the fence is unbothered.

By default the fence is built from the pair mass of **every occupied bin**,
not from the shortlist. If the shortlist happens to be entirely beads, its
quartiles are computed from beads, the IQR is enormous, and the fence stops
firing. Using the whole field as the reference guarantees ordinary field is in
the sample; it costs one `bincount`.

There is deliberately **no largest-*gap* fallback**. Because it searched the
whole sorted sample it happily split a field with two density modes and called
the upper mode fiducials; the effect-size floor refuses that, continuously.

**A minimum reference size** (`min_reference_bins`, default 16). An outlier
test needs a sample. With one or two bins in the field *everything* looks like
an outlier — a bead-free field that fits inside a single bin would otherwise be
reported as a fiducial, and the recommended mask would then delete the entire
dataset. Below the floor nothing is named and `report["insufficient_reference"]`
says so. Set it to 0 to opt out.

**A chance test**, `expected_false_bins` (see [API](#api)), as a necessary
condition. A log-space fence has no notion of how many bins it looked at, and
the maximum of tens of thousands of Poisson draws will sail past it. So a
candidate is also required to have a peak bin occupancy that a uniform field of
the same density would *not* be expected to reach by chance anywhere:

```
expected_false_bins = n_bins × P(Poisson(λ) ≥ peak bin count)
```

`n_bins` is the field's area divided by the bin area, so it does not move when
the histogram grid is shifted — the model describes the field, not where the
lines fall. `λ` and `n_bins` describe the same population: `λ` is a per-bin mean over
**all** bins of the field including empty ones, estimated from the bins outside
the proposal shortlist so beads cannot inflate it, and `n_bins` is the total
bin count. (Averaging over *occupied* bins would be zero-truncated — on a
sparse field it overestimates the rate by an order of magnitude and rejects
legitimate faint sources.) Reject if the result is at least
`max_expected_false_bins`, default 1.0.

**Localizations outside the field are discarded before anything else happens**,
so the histogram, the shortlist, the candidates and the fence reference all
describe the same field the trial count does. `report["n_locs_outside_field"]`
says how many rows that cost — which is the other half of why the field is a
required argument rather than a guess.

The shortlist is computed **twice**: once from the raw density, then again from
a background re-estimated outside the first shortlist. Without the second pass
a bright source inflates the background it is judged against and raises the cut
enough to hide a faint one — measured, a 30,000-localization source lifted the
cut from 15 to 23 and a real 45-localization source stopped being proposed.

**Confidence.** Spatial evidence is not the whole story, so acceptance is
graded. A candidate that passes everything above is `high` confidence and is
returned in `centers_nm` for automatic removal. One whose localizations arrive
in a few brief episodes far apart is `needs_review`: it is reported, with its
numbers, but **not returned and not removed**. Two 100-frame bursts at opposite
ends of a movie genuinely do have more cross-time pairs than a continuously
present bead — 1.0e8 against 3.1e7, measured — and no pair count can tell them
apart, so the decision is handed back rather than guessed.

The test is `temporal_coverage`: the source's background-subtracted temporal
entropy divided by the entropy a *contiguous* source of the same span would
show. A bead attached for only a fifth of the movie scores ~1 just like one
present throughout; two brief episodes score ~0.2; a single burst shorter than
one block scores 0, because a source that brief carries no evidence of
persistence at all. Below 0.5 goes to review.

That last case is why the undefined-span answer is 0 and not 1: an earlier
version divided by a non-positive expectation there and returned 1.0, so
making a 100-frame transient *brighter* moved it from `needs_review` to
automatic removal.

There is no localization-count floor and no maximum bead count.

## Limits worth knowing

**Resolution is one to two bins.** Two beads closer together than
`max_drift_nm` always come back as one centre — intrinsic to binning at that
scale, with no sub-bin information to split them. At two bins and beyond they
always come back as two. In between it depends on where the grid lines fall
relative to the pair: a bead split across two cells can be swallowed by a
stronger neighbour. Measured on synthetic pairs at four grid phases: 1 centre
at 0.6, 1.0 and 1.2 × bin; 1 or 2 at 1.6 ×; always 2 at 2.0 × and beyond.

**Under-binning duplicates, it does not invent.** With `max_drift_nm` set four
times too small, a real crop returned 33 centres for 17 beads — but 32 of the
33 were within 300 nm of a real bead. Passing a badly wrong `max_drift_nm`
splits clouds; it does not conjure sources.

**A short bright burst can still clear the fence.** The pair criterion demotes
it by two orders of magnitude relative to a bead with the same localization
count, and the two half-offset grids remove the grid-phase artefact entirely,
but the fence is relative: in an otherwise quiet field a burst is still the
biggest thing in sight. `frame_span` and `n_time_blocks_present` are in the
report so you can see it.

## What the report contains

| Key | Meaning |
| --- | --- |
| `accepted`, `rejected` | Per-candidate dicts, each list sorted by score |
| `score_threshold`, `min_effect_size` | Where the fence landed, and the floor under it |
| `reference_shortlist_fraction`, `fence_undefined_reason` | How much of the positive reference is on the shortlist, and why nothing was named if nothing was (`high_shortlist_fraction` or `no_separation`) |
| `n_reference_bins` | Bins the fence was placed against — only bins at least half inside the field, since a sliver holds a fraction of a bin's localizations and the square of that fraction in pairs |
| `n_entropy_blocks`, `entropy_block_frames` | The temporal grid `temporal_entropy` was measured on |
| `n_reference_bins`, `min_reference_bins`, `insufficient_reference` | The fence's sample size, and whether it was too small to proceed |
| `field_bounds_nm`, `n_locs_in_field`, `n_locs_outside_field` | The statistical field of view, and what fell outside it |
| `chance_occupancy`, `first_pass_occupancy` | The derived proposal cut, and the cut before the background was re-estimated |
| `bin_size_nm`, `n_occupied_bins`, `n_selected_bins`, `n_components` | The histogram and the sizes at each stage |
| `n_bins`, `background_rate_locs_per_bin`, `max_expected_false_bins` | Inputs to the chance test |
| `time_block_frames`, `n_time_blocks`, `minimum_persistence`, `effective_minimum_persistence`, `entropy_block_frames`, `n_entropy_blocks`, `pair_radius_nm`, `compute_fine_pair_mass`, `grid_offset_nm`, `merge_edges` | Settings, echoed, including the persistence actually achieved |

Each candidate dict carries:

| Key | Meaning |
| --- | --- |
| `center_nm` | Centre after the box mean shift |
| `n_locs`, `peak_bin_locs` | Window occupancy, and the fullest single grid bin |
| `pair_mass` | Cross-time complete-graph pair mass over the window — the default score |
| `pair_mass_strided` | The same off the subsample, as an estimator calibration |
| `pair_mass_fine`, `concentration` | KD-tree pair mass at `pair_radius_nm`, and its ratio to a uniform bin; `nan` unless `compute_fine_pair_mass` |
| `temporal_entropy` | Normalised entropy over a grid of **at least** 64 blocks, and never coarser than the scoring grid: 1.0 = spread evenly over the acquisition, low = a few episodes |
| `temporal_coverage` | That entropy over the entropy a contiguous source of the same span would show. ~1 for continuous presence however brief, ~0.2 for disconnected episodes |
| `confidence` | `high`, `needs_review` or `rejected` |
| `n_background_sectors` | Sectors of the background ring that were usable — 8 in the open field, as few as 2 at a corner, where the same estimate carries about twice the spread |
| `peak_bin_exposure` | That bin's share of a whole one; below 1 only on the boundary of the field, and the chance test is taken at it — so `expected_false_bins` for that candidate uses `background_rate_locs_per_bin` × this, not the bare rate |
| `n_bins`, `footprint_bins` | Bins in the merged component, and shortlisted bins in the candidate's own 3×3 neighbourhood. The second is what the compact rule reads, because it means the same thing with `merge_edges` on or off |
| `radius_p99_nm` | Cloud radius with the local background subtracted — **the removal radius**, and `fiducial_mask` takes one per bead |
| `window_frame_span`, `source_frame_span` | The raw extent of everything in the window, and where the source's own mass is — the second resolved to the entropy block and clamped to the acquisition — a handful of background localizations stretched a true 4000–4099 burst to (231, 7339), so the second is what a reviewer should be shown |
| `n_time_blocks_present` | How many scoring blocks the window touches |
| `subsample_stride`, `n_bins`, `grid_shift_nm` | Internals, for QC |
| `expected_false_bins` | Chance-test value |
| `reject_reason` | Anything not accepted: `below_fence`, `expected_by_chance`, `temporal_coverage` or `insufficient_reference` |

`fiducial_mask` accepts one radius per centre, so `radius_p99_nm` can be passed
straight through: preview, record and apply the same radii, and fall back to a
flat radius only for centres a hand review added or moved. A radius of zero is legal and means an exact coordinate match: a
perfectly stationary or heavily quantized source can measure zero, and masking
it is not an error.

The radius is **background-corrected**, and it has to be. The raw percentile of
every localization inside the measuring disc is not the source's radius:
background grows as `r²`, so for anything but a very bright source the outer
part of the disc dominates and the percentile lands on the disc boundary. On a
simulated bead of true radius ~30 nm the raw percentile reported 240 nm at 150
localizations — 7× in radius is 50× in area — and only converged above a few
thousand. So the source is measured inside a disc of **two bins** — the resolution
limit, so nothing closer is a separate source anyway — with the background
density estimated from the ring just beyond it, at two to three bins. Each
sector of that ring is normalised by its **observable** area, because two
things routinely make part of it unobservable: the edge of the field, and the
territory of a neighbouring candidate, which is masked out of both the counts
and the area.

A one-bin disc was measured to truncate real clouds. A bead whose cloud is
wider than `max_drift_nm` — the case the bin sweep says *detection* tolerates —
then reported a radius pinned at the cap while its own tail sat in the ring and
inflated the background, so removal silently left part of it behind: 83.8% of
the source removed at a cloud sigma of 120 nm. With a two-bin disc that is
98.7%, and 97.9% even at sigma 150. Measured after: 25–40 nm across 150 to 8,000 localizations,
against a true 29–33 nm.

That annulus is not always background either — another resolved fiducial two
bins away sits in it, and a plain mean density there ate the source: two beads
500 nm apart each reported 19 nm against a true 30 and left 17% of themselves
behind. So the density is the **median over eight angular sectors**, which one
contaminating direction cannot move. Measured after: 30.1–30.2 nm at 500 nm
separation, with 99% of both beads removed. On the real crop the measured radii removed
2.88 M localizations carrying 99.6% of the pair mass, against 2.90 M and
99.7% for a flat 250 nm — the same result for slightly less data thrown away.

## Measured behaviour

Real data — a 3.94 M-localization crop with seventeen hand-reviewed bead
centres:

| | |
| --- | --- |
| Recall | **17 / 17** |
| False positives | **0** |
| Median centre error | **13.9 nm** |
| Runtime | **3.6 s**, one CPU |
| Separation at the fence | 17× |

Robustness on the same data: full recall with no false positives at
`max_drift_nm` and at twice it. A bead-free dataset yields nothing at any of
three `max_drift_nm` values; a uniform random field yields nothing, with or
without a stray coordinate outlier; and a bead-free field with two density
modes yields nothing. Synthetically, beads placed exactly on grid lines are centred to 0.1 nm, and
beads are found down to 312 localizations against a 300 k background — the
effect-size floor costs the 78-localization rung that an unfloored fence
reached, which is the trade in the table above.

Against Picasso's own `imageprocess.find_fiducials` on the same crop, run in
the same environment: **17/17 with no false positives, against 0 found** — at
comparable cost, 3.1 s against 2.6 s. Picasso's finder keeps only picks holding
more than `0.8 × n_frames` localizations, and no bead in this 819,182-frame
crop reaches that. With the gate removed its detector does land on all 17, but
with 187 other spots alongside them and centres quantized to the camera pixel
grid (49.6 nm median error). The difference is selectivity, not speed.

## Using it with COMET

Nothing is automatic — the detection is a separate step you invoke, and COMET
is unchanged either way. The two normal uses:

```python
from comet import comet_run_kd
from comet_fiducials import find_fiducials, fiducial_mask

centers_nm, report = find_fiducials(locs, 250, field_bounds_nm=camera_fov_nm)
radii = [c["radius_p99_nm"] for c in report["accepted"]]
mask = fiducial_mask(locs, centers_nm, radii)

# anything spatially convincing but temporally not is here, not removed:
for item in report["needs_review"]:
    print(item["center_nm"], item["temporal_coverage"])

# (a) remove the beads so the structure drives the correction
drift = comet_run_kd(locs[~mask], segmentation_mode=2, segmentation_var=500,
                     max_drift_nm=250)

# (b) or keep only the beads and use them as the drift reference
drift = comet_run_kd(locs[mask], segmentation_mode=2, segmentation_var=500,
                     max_drift_nm=250)
```

## API

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

::: comet_fiducials.expected_false_bins
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true

::: comet_fiducials.outer_fence_threshold
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      show_signature: true
      separate_signature: true
