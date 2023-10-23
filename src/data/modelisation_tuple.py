from dataclasses import (
    dataclass,
)  # https://docs.python.org/fr/3/library/dataclasses.html
import pandas as pd


@dataclass
class modelisationTuple:
    """
    Holds together X and Y frames
    """

    train_X: pd.DataFrame
    train_y: pd.DataFrame
    test_X: pd.DataFrame
    test_y: pd.DataFrame
    scoring_X: pd.DataFrame
