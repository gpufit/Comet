import argparse
from comet.core.drift_optimizer import comet_run_kd
from comet.core.io_utils import (
    load_thunderstorm_csv,
    load_normal_molecule_set,
    save_dataset_as_thunderstorm_csv,
)


def main():
    parser = argparse.ArgumentParser(description="COMET drift correction CLI")

    parser.add_argument("--input", required=True, help="Path to input file (.csv or .h5)")
    parser.add_argument("--output", required=True, help="Path to save corrected localizations")
    parser.add_argument("--segmentation_mode", type=int, default=2, help="0: num windows, 1: locs/window, "
                                                                         "2: frames/window")
    parser.add_argument("--segmentation_var", type=int, required=True, help="Segmentation variable (depends on mode)")
    parser.add_argument("--initial_sigma_nm", type=float, default=100)
    parser.add_argument("--target_sigma_nm", type=float, default=1)
    parser.add_argument("--max_drift", type=float, default=None)
    parser.add_argument("--boxcar_width", type=int, default=1)
    parser.add_argument("--interpolation", choices=["cubic", "catmull-rom"], default="cubic")
    parser.add_argument("--format", choices=["csv", "h5"], required=True, help="Output file format")
    parser.add_argument("--display", action="store_true", help="Display intermediate results during processing")

    args = parser.parse_args()

    if args.input.endswith(".csv"):
        dataset = load_thunderstorm_csv(args.input)
    elif args.input.endswith(".h5"):
        dataset = load_normal_molecule_set(args.input)
    else:
        raise ValueError("Unsupported input format")

    drift, corrected = comet_run_kd(
        dataset=dataset,
        segmentation_mode=args.segmentation_mode,
        segmentation_var=args.segmentation_var,
        initial_sigma_nm=args.initial_sigma_nm,
        target_sigma_nm=args.target_sigma_nm,
        max_drift=args.max_drift,
        boxcar_width=args.boxcar_width,
        return_corrected_locs=True,
        interpolation_method=args.interpolation,
        display=args.display,
        save_corrected_locs=args.format == "h5",
    )

    if args.format == "csv":
        save_dataset_as_thunderstorm_csv(corrected, args.output)


if __name__ == "__main__":
    main()
