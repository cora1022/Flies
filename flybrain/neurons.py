"""Look up FlyWire neurons by cell type / side using the v783 annotations."""
from functools import lru_cache

import pandas as pd

from .brain import DATA_DIR

# Labellar sugar GRNs used in Shiu et al. 2024 (Fig. 1)
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = [720575940660219265, 720575940618238523]  # proboscis motor neuron (cell type CB0701)


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "flywire_annotations_783.tsv", sep="\t",
                       usecols=["root_id", "super_class", "cell_class", "cell_type",
                                "hemibrain_type", "side", "top_nt"],
                       dtype={"cell_type": str, "hemibrain_type": str})


def by_type(cell_type: str | list[str], side: str | None = None) -> list[int]:
    """Root IDs of neurons with the given cell type(s), optionally one side only."""
    types = [cell_type] if isinstance(cell_type, str) else cell_type
    a = annotations()
    m = a["cell_type"].isin(types) | a["hemibrain_type"].isin(types)
    if side is not None:
        m &= a["side"] == side
    return a.loc[m, "root_id"].tolist()
