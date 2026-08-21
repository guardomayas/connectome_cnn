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
    return pd.read_csv(_FILES[cell_class])


def get_dog_params(cell_class: str, celltype: str) -> dict:
    """Return DoG params for one input celltype, ready to pass into DoG_1D/DoG_2D."""
    table = load_input_table(cell_class)
    matches = table.loc[table["Celltype"] == celltype]

    if matches.empty:
        available = sorted(table["Celltype"].unique())
        raise ValueError(
            f"Unknown celltype {celltype!r} for cell_class {cell_class!r}, "
            f"expected one of {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous celltype {celltype!r} for cell_class {cell_class!r}: "
            f"{len(matches)} matching rows found"
        )

    row = matches.iloc[0]
    return {
        "sign": row["Sign"],
        "fwhm_c": row["FWHMcen(deg)"],
        "fwhm_s": row["FWHMsur(deg)"],
        "A_rel": row["Asur/Acen"],
    }