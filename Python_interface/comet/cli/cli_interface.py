import argparse
from comet.core.drift_optimizer import comet_run_kd
from comet.core.io_utils import (
    load_thunderstorm_csv,
    load_normal_molecule_set,
    correct_and_save_thunderstorm_csv,
    save_dataset_as_thunderstorm_csv,
    save_dataset_as_ms_h5,
)


def main():
    parser = argparse.ArgumentParser(description="COMET drift correction CLI")

    parser.add_argument("--input", "-i", required=True, help="Path to input file (.csv or .h5)")
    parser.add_argument("--output", "-o",  required=True, help="Path to save corrected localizations")
    parser.add_argument("--segmentation_mode", "-sm", type=int, default=2, help="0: num windows, 1: locs/window, "
                                                                         "2: frames/window")
    parser.add_argument("--segmentation_var", "-sv", type=int, required=True, help="Segmentation variable (depends on mode)")
    parser.add_argument("--initial_sigma_nm", "-isig", type=float, default=None,
                        help="Initial sigma (nm). Defaults to max_drift_nm/3 when omitted.")
    parser.add_argument("--target_sigma_nm", "-tsig", type=float, default=30.0)
    parser.add_argument("--max_drift_nm", "-d", type=float, default=300.0)
    parser.add_argument("--boxcar_width", type=int, default=1)
    parser.add_argument("--interpolation", choices=["cubic", "catmull-rom"], default="cubic")
    parser.add_argument("--format", choices=["csv", "h5"], help="Output file format", default="csv")
    parser.add_argument("--display", action="store_true", help="Display intermediate results during processing")
    parser.add_argument("--max_locs_per_segment", "-mlpseg", type=int, default=None)
    parser.add_argument("--mode", choices=["cuda", "cuda_qc", "cpu", "torch", "torch_qc"], default="cuda")
    args = parser.parse_args()

    input_is_csv = args.input.lower().endswith(".csv")
    input_is_h5 = args.input.lower().endswith(".h5")
    if input_is_csv:
        dataset = load_thunderstorm_csv(args.input)
    elif input_is_h5:
        dataset = load_normal_molecule_set(args.input)
    else:
        raise ValueError("Unsupported input format; expected .csv or .h5.")

    drift, corrected = comet_run_kd(
        dataset=dataset,
        segmentation_mode=args.segmentation_mode,
        segmentation_var=args.segmentation_var,
        initial_sigma_nm=args.initial_sigma_nm,
        target_sigma_nm=args.target_sigma_nm,
        max_drift_nm=args.max_drift_nm,
        max_locs_per_segment=args.max_locs_per_segment,
        boxcar_width=args.boxcar_width,
        return_corrected_locs=True,
        interpolation_method=args.interpolation,
        mode=args.mode,
        display=args.display,
    )

    if args.format == "csv":
        if input_is_csv:
            # Preserves the full set of columns from the original ThunderSTORM CSV.
            correct_and_save_thunderstorm_csv(drift, args.input, args.output)
        else:
            save_dataset_as_thunderstorm_csv(corrected, args.output)
    else:
        save_dataset_as_ms_h5(corrected[:, :3], corrected[:, 3].astype(int),
                              pixelsize_nm=160, filename=args.output)


if __name__ == "__main__":
    main()
