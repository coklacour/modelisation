##########################################################################
#                                Packages                                #
##########################################################################

import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.utils import resample

##########################################################################
#                                 Script                                 #
##########################################################################

class BalanceMixin:
    """
    Removing a portion of the observations of the majority class to prevent its signal from dominating
    """

    def __init__(self):
        super().__init__()
 
    def _balance(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Rebalance an X dataframe by undersampling the majority class

        Args:
            X (pd.DataFrame): The dataframe to be downsampled with respect to y.
            y (np.array): The one-hot target vector to use to downsample X.

        Returns:
            pd.DataFrame, np.ndarray: Downsampled X and y.
        """

        X_majority = X[y == 0]
        X_minority = X[y == 1]

        if len(X_minority) == 0:
            raise ValueError(
                "Minority class must have at least one occurency for balancing to work."
            )

        X_majority_downsampled = resample(
            X_majority, replace=False, n_samples=X_minority.shape[0]
        )  # reproducible results

        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([X_majority_downsampled, X_minority])
        zeros = np.zeros(len(X_majority_downsampled))
        ones = np.ones(len(X_minority))
        y = np.concatenate([zeros, ones])

        return df_downsampled, y
