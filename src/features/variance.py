# noq##########################################################################
#                                Packages                                #
##########################################################################

import pandas as pd
import numpy as np

import datapane as dp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from src.features.balance import BalanceMixin
from utils.reportable import ReportableAbstract

##########################################################################
#                                 Script                                 #
##########################################################################


class NearZeroVar_filter(
    BaseEstimator, TransformerMixin, BalanceMixin, ReportableAbstract
):
    """
    Step to remove all features with very frequent values and low variability from the dataset.
    """

    GROUP_NAME = "Near-Zero-Variance filter"

    _HEADER = """
    ## Near-Zero-Variance filter

    _Removes uninformative columns before training the model, based on the number of possible values observed. Works with both quantitative and qualitative variables_

    ### Details of the selection procedure
    """

    def __init__(
        self,
        frequency_ratio: float = 95 / 5,
        unique_cut: float = 0.05,
        equisample: bool = True,
    ):
        """
        Args:
            frequency_ratio (float, optional): The cutoff for the ratio of the most common value to the second most common value. Defaults to 95/5
            unique_cut (float, optional): the cutoff for the percentage of distinct values out of the number of total samples. Defaults 0.05
            equisample (boolean, optional): Schould the algorithm equisample the two classes before estimating the variance ?. Defaults to True.
        """

        super().__init__()
        self.frequency_ratio = frequency_ratio
        self.unique_cut = unique_cut
        self.equisample = equisample

        # Placeholder
        self._cols = []
        self._removed = []

    def get_feature_names_out(self, input_features=None):
        return self._cols

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Extract the features that we will keep

        Args:
            X (pd.DataFrame): the dataframe
            y (np.ndarray): The target vector
        """

        if self.equisample and y is not None:
            X, _ = self._balance(X, y)

        def _freq_cut(serie) -> bool:
            """
            Check if the frequency of the most prevalent value over the second most frequent value is enough

            Args:
                serie (pd.Series): The numeric pandas serie to invstigate
            """

            try:
                F, S = serie.value_counts().iloc[0:2]
            except ValueError:
                return True  # ie, the variable doesn't

            return F / S >= self.frequency_ratio

        unique_cut = lambda x: len(x.unique()) / len(x) < self.unique_cut  # noqa: E731

        # Remove the NZV
        zero_var = pd.Series(X.apply(_freq_cut) & X.apply(unique_cut)).to_frame("drop_")
        self._cols = zero_var[zero_var.drop_ is False].index.to_list()  # type: ignore
        self._removed = zero_var[zero_var.drop_ is True].index.to_list()  # type: ignore

        return self

    def transform(self, X: pd.DataFrame):
        """
        Remove the near zero variance features from the X dataframe.

        Args:
            X (pd.DataFrame): the dataframe
        """

        return X[self._cols]

    def get_report_group(self):
        """
        Generates a Datapane report group to describe the Near-Zero-Variance filtering step.

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


class LowVar_Filter(BaseEstimator, TransformerMixin, BalanceMixin, ReportableAbstract):
    """
    Step to remove all features with low-variance
    """

    GROUP_NAME = "Low Variance filter"

    _HEADER = """
    ## Low variance filter

    _Removes uninformative columns before training the model, based on too low a variance._

    ### Details of the selection procedure
    """

    def __init__(self, threshold: float = 0.01, equisample: bool = True):
        """
        Args:
            threshold (float, optional): The minimal variance a column must have to be considered. Defaults to 0.01
            equisample (boolean, optional): A Boolean indicating whether class balancing should be performed before correlation calculation. Defaults to True.
        """

        super().__init__()
        self.threshold = threshold
        self.equisample = equisample

        # Placeholders
        self._cols = []
        self._removed = []

    def get_feature_names_out(self, input_features=None):
        return self._cols

    def fit(self, X: pd.DataFrame, y=None):
        """
        Extract the features that we will keep

        Args:
            X (pd.DataFrame): The DataFrame to remove columns from.
        """

        if self.equisample is True and y is not None:
            X, _ = self._balance(X, y)

        # Eliminate features whose variance is less than self.threshold
        selector = VarianceThreshold(threshold=self.threshold).fit(X)
        self._cols = X.columns[selector.get_support()]
        self._removed = list(set(X.columns) - set(self._cols))

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Remove features with too low a variance from the X data frame.

        Args:
            X (pd.DataFrame): The dataframe to filter columns for.
        """

        return X[self._cols]

    def get_report_group(self):
        """
        Generates a Datapane report group to describe the low-variance filtering step.

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
