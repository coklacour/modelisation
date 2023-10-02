##########################################################################
#                                Packages                                #
##########################################################################

from typing import Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

##########################################################################
#                                 Script                                 #
##########################################################################

def metric_pr_auc(y: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        y (np.ndarray): True binary labels {0, 1}
        probs (np.ndarray): Probability estimates of the positive class {1}

    Returns:
        float: The PR AUC
    """

    precision, recall, _ = precision_recall_curve(y, probs)

    return auc(recall, precision)
    
def metric_pr_f_beta(y: np.ndarray, probs: np.ndarray, beta: float = 1.0) -> float:
    """
    Compute the F-beta score for a given true vector 'y' and a vector of probabilities 'probs'
    
    Args:
        y (np.ndarray): The true vector array
        probs (np.ndarray): Target scores, can either be probability estimates of the positive class, or non-thresholded measure of decisions
        beta (float, optional): The harmonic weight between the precision and recall. Defaults to 1.0.

    Returns:
        float: The F-beta Score
    """
    
    # we are aware that there may be divisions by 0. We deal with this situation and do not wish to see any warnings
    np.seterr(divide='ignore', invalid='ignore')


    precision, recall, _ = precision_recall_curve(y, probs)

    # handles divide-by-zero
    denominator = beta ** 2 * precision + recall
    mask = denominator == 0.0

    fscore = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

    # On zero-division, sets the corresponding result elements equal to 0
    fscore[mask]=0

    return np.max(fscore)

def get_threshold_best_pr_fbeta(y: np.ndarray, probs: np.ndarray, beta: float = 1.0) -> Tuple[float, float]:
    """
    Compute the largest F-beta score achievable for a given true vector 'y' and a corresponding vector of probabilities 'probs'
    
    Args:
        y (np.ndarray): The true vector array
        probs (np.ndarray): Target scores, can either be probability estimates of the positive class, or non-thresholded measure of decisions
        beta (float, optional): The harmonic weight between the precision and recall. Defaults to 1.0.

    Returns:
        Tuple[float, float]: [the largest F-beta score, the threshold for the highest F-score is reached]
    """
    
    # we are aware that there may be divisions by 0. We deal with this situation and do not wish to see any warnings
    np.seterr(divide='ignore', invalid='ignore')


    precision, recall, thresholds = precision_recall_curve(y, probs)

    # handles divide-by-zero
    denominator = beta ** 2 * precision + recall
    mask = denominator == 0.0

    fscore = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

    # On zero-division, sets the corresponding result elements equal to 0
    fscore[mask]=0

    idx = np.argmax(fscore)
    
    return fscore[idx], thresholds[idx]