import os
import numpy as np
from comet.core.io_utils import (
    load_thunderstorm_csv,
    load_normal_molecule_set,
    save_dataset_as_thunderstorm_csv,
    save_dataset_in_ms_format_h5
)


def data_path(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..\\..", "data", filename))


def test_load_and_save_thunderstorm_csv(tmp_path):
    csv_file = data_path("npc_n96_3d_prec_2nm_2_loc_per_frm_000.csv")

    locs = load_thunderstorm_csv(csv_file)

    assert isinstance(locs, np.ndarray)
    assert locs.shape[1] in [3, 4]
    assert np.all(locs[:, :2] >= 0)

    out_path = tmp_path / "test_output.csv"
    save_dataset_as_thunderstorm_csv(locs, str(out_path))
    assert out_path.exists()


def test_load_and_save_molecule_set(tmp_path):
    h5_file = data_path("npc_n96_3d_prec_2nm_2_loc_per_frm_000.molecule_set.h5")
    locs = load_normal_molecule_set(h5_file)

    assert isinstance(locs, np.ndarray)
    assert locs.shape[1] == 4

    padded_locs = np.pad(locs, ((0, 0), (0, 5)), constant_values=1)
    out_path = tmp_path / "test_output.h5"
    save_dataset_in_ms_format_h5(padded_locs, str(out_path))
    assert out_path.exists()
