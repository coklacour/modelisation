#############################################################################
#                                 Packages                                  #
#############################################################################

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import optuna

from src.models.binary_classification.evaluate.scorers import (
    _scorer_aucpr_non_degenerated,
    _DegeneratedF1Error,
)
from src.models.binary_classification.evaluate.metrics import (
    get_threshold_best_pr_fbeta,
)

from utils.logger import get_logger


#############################################################################
#                                 Scripts                                   #
#############################################################################

LOGGER = get_logger("LOGISTIC_autotunning")

# Metric for optimization
_OPTIM_METRIC = {"aucpr": _scorer_aucpr_non_degenerated}

# Add constant parameters not supported by optuna
_TRAINING_CONST_PARAMS = {
    "solver": "liblinear",
    "max_iter": 5000,
}


def _logistic_search(X, y, optim_metric_strategy="aucpr", n_trials=2):
    """
    Optimize a logistic model
    """

    # Fetch the objective function to use
    scorer = _OPTIM_METRIC.get(optim_metric_strategy, None)
    if not scorer:
        raise NotImplementedError(
            f"The {optim_metric_strategy} is not implemented. Choose from {list(_OPTIM_METRIC.keys())}"
        )

    # Use the trial object to suggest values for C and penalties
    def _(trial):
        param = {
            "C": trial.suggest_float("C", 1e-8, 3.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }

        param.update(**_TRAINING_CONST_PARAMS)

        m = LogisticRegression(**param)
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        try:
            results = cross_val_score(
                m, X, y, cv=kfold, scoring=scorer, error_score="raise"
            )
            r = results.mean()
        except _DegeneratedF1Error as e:
            LOGGER.warn(
                f"Discarding Optuna trial because of a degenerated f1. Trial params: {param}"
            )
            LOGGER.exception(e)
            r = np.nan

        return r

    study = optuna.create_study(direction="maximize")
    study.optimize(_, n_trials=n_trials, timeout=3600)  # Spend one hour on each model

    try:
        params = study.best_trial.params
    except ValueError:
        params = {}

    return params


class LogisticAutoTuning(BaseEstimator, ClassifierMixin):
    """
    Returns a Logistic Classification model tuning with Optuna
    """

    def __init__(
        self,
        optuna_trials: int = 10,
        model_params=None,
        optim_metric_strategy="aucpr",
    ):
        """
        Return a new Logistic autotuner.

        Args:
            optuna_trials (int, optional): The number of trials for the optimsation pass. Defaults to 10.
            model_params (dictionary, optionnal) : Hyperparameters that allowed to obtain the best performance according to optim_metric_strategy. Defaults to None.
            optim_metric_strategy (string): the objective function for Optuna to maximize. Defaults to aucpr,
        """

        super().__init__()

        self.optuna_trials = optuna_trials
        self.model_params = model_params
        self.optim_metric_strategy = optim_metric_strategy

        # Placeholder
        self._is_fitted = None
        self._threshold = None
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Search for the best parameters of a Logistic.

        Args:
            X (pd.DataFrame): The dataframe of input features to use to predict y.
            y (np.ndarray): The y target to be predicted.
        """

        LOGGER.info("Train a logistic classifier")

        # Sanity checks
        self.classes_ = unique_labels(y)

        # Search for the best hyperparameters or use the provided ones
        if self.model_params:
            params = self.model_params
        else:
            params = self.model_params = _logistic_search(
                X,
                y,
                optim_metric_strategy=self.optim_metric_strategy,
                n_trials=self.optuna_trials,
            )

        params.update(**_TRAINING_CONST_PARAMS)
        self._model = LogisticRegression(**params).fit(X, y)

        # Obtain the threshold for the best f1 score
        probs = self._model.predict_proba(X)[:, 1]
        _, threshold = get_threshold_best_pr_fbeta(y, probs)
        self._threshold = threshold

        # Toogle fit indicator
        self._is_fitted = True

        return self

    @property
    def model(self):
        return self._model

    def predict(self, X: pd.DataFrame):
        """
        Return prediction targets of a new set of data.

        Args:
            X (pd.DataFrame): The new set of data to run the prediction on.
        """

        return self.predict_proba(X)[:, 1] > self._threshold

    def predict_proba(self, X: pd.DataFrame):
        """
        Returns for each new observation the probability that it belongs to each target.

        Args:
            X (pd.DataFrame): The new set of data to run the prediction on.
        """

        return self._model.predict_proba(X)
