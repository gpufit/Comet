from __future__ import annotations

import sys
import traceback
import numpy as np
from matplotlib import pyplot as plt
from comet.core.drift_optimizer import comet_run_kd
import argparse


def simulate_dataset(
    n_points=1000,         # base template molecules
    T=500,                 # timesteps / frames
    locs_per_frame=120,    # localizations per frame
    seed=123,
):
    rng = np.random.default_rng(seed)

    # Base coordinates (nm), spread in a 1000 nm cube
    gt_coords = rng.random((n_points, 3)) * 1000.0  # (N,3)

    # Time axis
    timesteps = np.arange(T)

    # 1D base drift waveform (nm / frame); then create 3D with cheap phase/scale offsets
    base = (3*np.sin(timesteps / 20.0 + 0.37) + 1.25 * np.cos(timesteps / 6.0 + 1.2) + timesteps/25) * 10.0
    dx = base
    dy = 0.7 * base
    dz = -0.5 * base

    # Per-frame drift (nm)
    gt_drift_nm = np.column_stack([dx, dy, dz])  # (T,3)

    # Vectorized localization generation
    M = T * locs_per_frame
    frame_idx = np.repeat(np.arange(T, dtype=np.int32), locs_per_frame)
    rng.shuffle(frame_idx)

    base_idx = rng.integers(0, n_points, size=M)
    base_xyz = gt_coords[base_idx]        # (M,3)
    drift_xyz = gt_drift_nm[frame_idx]    # (M,3)
    xyz = base_xyz + drift_xyz            # (M,3)

    # Pack into (N,4): x,y,z,frame
    locs = np.empty((M, 4), dtype=np.float32)
    locs[:, :3] = xyz.astype(np.float32)
    locs[:, 3] = frame_idx.astype(np.float32)
    return locs, gt_drift_nm


def run_mock_comet(dataset):
    drift_cuda, _ = comet_run_kd(
        dataset=dataset,
        segmentation_mode=2,
        segmentation_var=2,
        initial_sigma_nm=120,
        max_drift=100,
        target_sigma_nm=10,
        drift_max_bound_factor=2,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        force_cpu=False,
        display=False
    )
    return drift_cuda[:, :3]


def _check_env() -> None:
    # super-light import sanity; keep it quiet & fast
    print("== COMET self-test ==")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import numpy  # noqa: F401
        print("OK   numpy")
    except Exception as e:
        print("FAIL numpy:", e)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot GT vs estimated drift")
    args = parser.parse_args()

    _check_env()

    try:
        dataset, gt = simulate_dataset()
        est = run_mock_comet(dataset)

        m = min(len(gt), len(est))
        if m < 5:
            raise AssertionError("Too few frames for a meaningful check (need ≥5).")

        # your offset-invariant metric: mean std of residuals per axis
        err_nm = np.mean(np.std(est[:m] - gt[:m], axis=0))
        print(f"Drift RMSE (nm): {err_nm:.2f}")  # (label kept as-is per your code)

        # optional plotting
        if args.plot:
            fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            labels = ("x", "y", "z")
            for i, ax in enumerate(axes):
                ax.plot(gt[:m, i], label=f"GT {labels[i]}")
                ax.plot(est[:m, i], "--", label=f"EST {labels[i]}")
                ax.set_ylabel("nm")
                ax.legend()
            axes[-1].set_xlabel("Frame")
            fig.suptitle("COMET self-test: GT vs EST drift")
            plt.tight_layout()
            plt.show()

        # single fixed threshold; tighten later once stable
        THRESHOLD_NM = 1.0
        if np.isfinite(err_nm) and err_nm < THRESHOLD_NM:
            print("✅ PASSED")
            return 0
        else:
            print(f"❌ FAILED: {err_nm:.2f} ≥ {THRESHOLD_NM} nm")
            return 2

    except AssertionError as ae:
        print("❌ Shape/consistency error:", ae)
        return 3
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print("❌ Unhandled error:")
        print(e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())