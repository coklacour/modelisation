#############################################################################
#                                 Packages                                  #
#############################################################################

import numpy as np
from src.models.binary_classification.evaluate import metric_pr_f_beta, metric_pr_auc


#############################################################################
#                                 Scripts                                   #
#############################################################################


class _DegeneratedF1Error(Exception):
    """Sentinel error used to abort an optuna trial when a f1 score is degenerated"""

    pass


def _scorer_aucpr_non_degenerated(estimator, X, y, scoring_weight_f1: float = 0):
    """
    Compute the AUC pr score

    Args:
        estimator (object): estimator that has been fitted to the training data. This estimator must have a `predict_proba` method for calculating prediction probabilities
        X (pd.DataFrame): the design matrix
        y (np.ndarray): The true vector array
        scoring_weight_f1 (float): weight that defines compromise between the f1 score and the AUC score in the objective function. Defaults to 0.

    Returns:
        float: The F-beta Score
    """

    # Extract both probability
    y_hat_both = estimator.predict_proba(X)

    # Compute both of the f1 score
    max_f1_0 = metric_pr_f_beta(1 - y, y_hat_both[:, 0])
    max_f1_1 = metric_pr_f_beta(y, y_hat_both[:, 1])

    # Check that both f1 score are semi-definite
    is_nan = np.isnan(max_f1_0) or np.isnan(max_f1_1)
    is_zeroes = max_f1_0 == 0 or max_f1_1 == 0
    if is_nan or is_zeroes:
        raise _DegeneratedF1Error()

    # compute the actual value
    y_hat = y_hat_both[:, 1]

    # The objective function is a compromise between the f1 score and the AUC score : the AUC score capture the global distribution of the score while the f1 score capture the local reported metrics
    return (1 - scoring_weight_f1) * metric_pr_auc(y, y_hat) + (
        scoring_weight_f1 * max_f1_1
    )


def _eval_aucpr_non_degenerated(predt, dtrain):
    """
    The function schould implements the same underlying logic than it counterpart '_scorer_aucpr_non_degenerated'.

    Note:
    - I don't use xgboost.cv becauseXGBoost supports only k-fold cross validation
    - I don't use sklearn.gridsearch. Pruning of trials that terminates unpromising trials will be preferred, so that computational time can be used for trials that show more potential.

    So the eval_metric / scoring parameters of the xgboost schould never be called.
    This function is just a placeholder acting as a reminder : the function schould be implemented if I were for switching to sklearn.gridsearch instead of optuna.

    Raises:
        NotImplementedError: The function is not currently implemented
    """

    raise NotImplementedError()
