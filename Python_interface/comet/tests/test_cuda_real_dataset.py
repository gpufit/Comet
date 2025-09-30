import numpy as np
from comet.core.io_utils import load_thunderstorm_csv
from comet.core.drift_optimizer import comet_run_kd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # go 2 levels up from comet/tests
DATA_DIR = ROOT / "test_dataset"


def data_path(filename: str) -> str:
    return str(DATA_DIR / filename)


def test_cuda_on_real_dataset(plot=True):
    csv_file = data_path("test_dataset.csv")
    dataset = load_thunderstorm_csv(csv_file)

    drift_cuda, _ = comet_run_kd(
        dataset=dataset,
        segmentation_mode=2,
        segmentation_var=50,
        initial_sigma_nm=40,
        max_drift=100,
        target_sigma_nm=10,
        drift_max_bound_factor=2,
        boxcar_width=1,
        return_corrected_locs=True,
        interpolation_method='cubic',
        force_cpu=False,
        display=False
    )

    if plot:
        plt.figure()
        plt.plot(drift_cuda[:, 0], label="X")
        plt.plot(drift_cuda[:, 1], label="Y")
        plt.plot(drift_cuda[:, 2], label="Z")
        plt.legend()
        plt.title("CUDA Drift Correction Output")
        plt.xlabel("Frame")
        plt.ylabel("Drift [nm]")
        plt.tight_layout()
        plt.show()

    assert drift_cuda.shape[1] == 4
    assert not np.isnan(drift_cuda).any(), "Drift output contains NaNs"


if __name__ == "__main__":
    test_cuda_on_real_dataset(plot=True)
