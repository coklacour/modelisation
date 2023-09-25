##########################################################################
#                                Packages                                #
##########################################################################

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

##########################################################################
#                                 Script                                 #
##########################################################################


class Passthrough(BaseEstimator, TransformerMixin):
    """
    Step to pass features through untransformed.
    Implemented for consistency with other transformers.
    """

    def __init__(self):
        """
        Return a new Passthrough object
        """

        super().__init__()

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
            y (np.ndarray): The target vector.
        """

        self._cols = X.columns.tolist()

        return self
    
    def transform(self, X: pd.DataFrame):
        """
        Args:
            X (pd.DataFrame): the dataframe
        """

        return X[self._cols]
