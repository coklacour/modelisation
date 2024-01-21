#############################################################################
#                                 Packages                                  #
#############################################################################

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score, StratifiedKFold

import optuna
from xgboost import XGBClassifier  # type: ignore

from src.models.binary_classification.evaluate.scorers import (
    _scorer_aucpr_non_degenerated,
    _eval_aucpr_non_degenerated,
    _DegeneratedF1Error,
)
from src.models.binary_classification.evaluate.metrics import (
    get_threshold_best_pr_fbeta,
)

from utils.logger import get_logger

#############################################################################
#                                 Scripts                                   #
#############################################################################


LOGGER = get_logger("XGBOOST_autotunning")

# Metric for optimization
_OPTIM_METRIC = {"aucpr": _scorer_aucpr_non_degenerated}

# To copte with the non-support by optuna of constant params
_TRAINING_CONST_PARAMS = {
    "verbosity": 0,
    "objective": "binary:logistic",
    "eval_metric": _eval_aucpr_non_degenerated,  # "aucpr"
    "scoring": _scorer_aucpr_non_degenerated,
}


def _xgboost_search(X, y, optim_metric_strategy="aucpr", n_trials=2):
    """
    Optimize an xgboost model
    """

    # Fetch the objective function to use
    scorer = _OPTIM_METRIC.get(optim_metric_strategy, None)
    if not scorer:
        raise NotImplementedError(
            f"The {optim_metric_strategy} is not implemented. Choose from {list(_OPTIM_METRIC.keys())}"
        )

    def _(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "lambda": trial.suggest_float("lambda", 1e-8, 3.0),
            "alpha": trial.suggest_float("alpha", 1e-8, 3.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 30, step=1),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 300),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
        }

        param.update(**_TRAINING_CONST_PARAMS)

        # Create an XGBClassifier model with the current hyperparameters
        m = XGBClassifier(
            **param
            # , callbacks=callbacks
        )

        # Define the cross-validation strategy
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        try:
            results = cross_val_score(
                m, X, y, cv=kfold, scoring=scorer, error_score="raise"
            )
            r = results.mean()
        except _DegeneratedF1Error:
            LOGGER.warn(
                f"Discarding Optuna trial because of a degenerated f1. Trial params: {param}"
            )
            r = np.nan

        return r

    study = optuna.create_study(direction="maximize")
    study.optimize(_, n_trials=n_trials, timeout=3600)  # Spend one hour on each model

    try:
        params = study.best_trial.params
    except ValueError:
        params = {}

    return params


class XGBoostAutoTuning(BaseEstimator, ClassifierMixin):
    """
    Returns a XGBoost Classification model tuning with Optuna
    """

    def __init__(
        self,
        optuna_trials: int = 10,
        model_params=None,
        optim_metric_strategy="aucpr",
    ):
        """
        Return a new XGBoost autotunner.

        Args:
            optuna_trials (int, optional): The number of trials for the optimsation pass. Defaults to 10.
            model_params () = . Defaults to None,
            optim_metric_strategy (string): the objective function for Optuna to maximize. Defaults to aucpr,
        """

        super().__init__()

        self.optuna_trials = optuna_trials
        self.model_params = model_params
        self.optim_metric_strategy = optim_metric_strategy

        # Placeholder
        self._fitted_model = None
        self._threshold = None
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Search for the best parameters of a XGBoost.

        Args:
            X (pd.DataFrame): The dataframe of input features to use to predict y.
            y (np.ndarray): The y target to be predicted.
        """

        # Sanity checks
        self.classes_ = unique_labels(y)

        # Maybe search for the best parameters, or use the provided ones
        if self.model_params:
            params = self.model_params
        else:
            params = self.model_params = _xgboost_search(
                X,
                y,
                optim_metric_strategy=self.optim_metric_strategy,
                n_trials=self.optuna_trials,
            )

        params.update(**_TRAINING_CONST_PARAMS)
        self._model = XGBClassifier(**params).fit(X, y)

        # Calibrate for the threshold for the best f1 score
        probs = self._model.predict_proba(X)[:, 1]
        _, threshold = get_threshold_best_pr_fbeta(y, probs)
        self._threshold = threshold

        # Toogle fit indicator
        self._fitted_model = True

        return self

    @property
    def model(self):
        return self._model

    def predict(self, X: pd.DataFrame):
        """
        Apply the model on a new dataset.

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
