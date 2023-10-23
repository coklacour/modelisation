##########################################################################
#                                Packages                                #
##########################################################################

import pandas as pd
import numpy as np

import datapane as dp

from sklearn.base import BaseEstimator, TransformerMixin
from src.features.balance import BalanceMixin
from utils.reportable import ReportableAbstract

##########################################################################
#                                 Script                                 #
##########################################################################


class HighCorrelation_filter(
    BaseEstimator, TransformerMixin, BalanceMixin, ReportableAbstract
):
    """
    Step to remove highly correlated features from the dataset.
    """

    GROUP_NAME = "High-Correlation filter"

    _HEADER = """
    ## High Correlation filter

    _Remove features that exhibit a high degree of linear correlation with other features before the models training._

    ### List of removed variables
    """

    def __init__(
        self,
        threshold: float = 0.95,
        equisample: bool = True,
    ):
        """
        Args:
            threshold (float, optional): correlation threshold above which features are considered highly correlated. Defaults to 0.95
            equisample (boolean, optional): A Boolean indicating whether class balancing should be performed before correlation calculation. Defaults to True.
        """

        super().__init__()
        self.threshold = threshold
        self.equisample = equisample

        # Placeholder
        self._cols = []
        self._removed = []

    def get_feature_names_out(self, input_features=None):
        return self._cols

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Extract the features that we will keep.

        Args:
            X (pd.DataFrame): the dataframe to remove the too correlated features form
            y (np.ndarray): The target vector.
        """

        if self.equisample and y is not None:
            X, _ = self._balance(X, y)

        # Compute the correlation
        # To Do: Isolate quantitative features and compute their correlations
        corr = X.corr().abs()

        # Extract the TRIU coeffs
        U = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_remove = tuple(c for c in U.columns if any(U[c] > self.threshold))
        self._cols = [c for c in X.columns if c not in to_remove]
        self._removed = [c for c in X.columns if c in to_remove]

        return self

    def transform(self, X: pd.DataFrame):
        """
        Remove the too-correlated features from the X dataframe.

        Args:
            X (pd.DataFrame): the dataframe
        """

        return X[self._cols]

    def get_report_group(self):
        """
        Generate a Datapane report group to describe the HighCorrelation filter step.

        Returns:
            dp.Group: The group to be wrapped in the report.
        """

        return dp.Group(
            self._HEADER,
            dp.Group(
                dp.BigNumber(
                    heading="Number of removed variables",
                    value=f"{len(self._removed)}",
                ),
                dp.BigNumber(
                    heading="Number of keeped variables", value=f"{len(self._cols)}"
                ),
                columns=2,
            ),
        )
