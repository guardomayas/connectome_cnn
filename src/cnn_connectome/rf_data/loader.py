from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent

_FILES = {
    "T4": _DATA_DIR / "T4_inputs.csv",
    "T5": _DATA_DIR / "T5_inputs.csv",
}


def load_input_table(cell_class: str) -> pd.DataFrame:
    """Load the fitted RF/temporal-filter params for a T4 or T5 input celltype table."""
    if cell_class not in _FILES:
        raise ValueError(f"Unknown cell_class {cell_class!r}, expected one of {list(_FILES)}")
    return pd.read_csv(_FILES[cell_class]).set_index("Celltype")


def get_dog_params(cell_class: str, celltype: str) -> dict:
    """Return DoG params for one input celltype, ready to pass into DoG_1D/DoG_2D."""
    row = load_input_table(cell_class).loc[celltype]
    return {
        "sign": row["Sign"],
        "fwhm_c": row["FWHMcen(deg)"],
        "fwhm_s": row["FWHMsur(deg)"],
        "A_rel": row["Asur/Acen"],
    }
