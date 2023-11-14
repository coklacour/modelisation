##########################################################################
#                                Packages                                #
##########################################################################

import pandas as pd
import numpy as np


from functools import reduce
import altair as alt
from math import log
from sklearn.metrics import auc, precision_recall_curve


##########################################################################
#                                 Script                                 #
##########################################################################


def plot_iso_f_beta(beta: float = 1.0):
    """
    This function creates a canvas with F_beta score iso-lines (in which the F_beta score maintains the same value).

    Args:
        beta (float, optional): The harmonic weight between the precision and recall. Defaults to 1.0.
    """

    f_scores = np.linspace(0.2, 0.8, num=4)

    canevas = []
    for f_score in f_scores:
        recall = np.linspace(0.01, 1)  # Generates 50 samples by default
        precision = f_score * recall / ((1 + beta**2) * recall - f_score * beta**2)

        iso_f_beta = pd.DataFrame(
            {
                "precision": precision[(precision >= 0) & (precision <= 1)],
                "recall": recall[(precision >= 0) & (precision <= 1)],
            }
        )

        g = (
            alt.Chart(iso_f_beta)
            .encode(
                x=alt.X("recall", title="Recall"),
                y=alt.Y("precision", title="Precision"),
            )
            .mark_line(color="grey", size=1, opacity=0.5)
        )

        text = (
            alt.Chart({"values": [{"x": 0.9, "y": f_score - 0.05}]})
            .mark_text(
                text=f"iso-f1 {f_score*100:.0f}%", color="grey", size=10, opacity=0.8
            )
            .encode(x="x:Q", y="y:Q")
        )

        out = g + text
        canevas.append(out)

    # reduce(fun,seq) : apply 'fun' to all of the list 'seq' elements
    canevas = reduce(lambda x, y: x + y, canevas)

    return canevas


def plot_pr_curve(y: pd.DataFrame, probs: np.ndarray, color: str = "black"):
    """
    Compute the PR curve for a given set of labels 'y' and a vector of probabilities 'probs'.
    The function display the PR-Curve as well as the Unachievable PR-curve and the normalized PR-curve as defined in https://arxiv.org/abs/1206.4667

    Args:
        y (pd.DataFrame): The binary vector of truth
        probs (np.ndarray): The vector of probabilities
        color (str): Color of the PR curve line
    """

    # Compute the PR curve for the estimaton
    precision, recall, _ = precision_recall_curve(y, probs)
    auc_ = auc(recall, precision)
    pr = pd.DataFrame({"precision": precision, "recall": recall})

    # Extract the natural prevalence
    pi = np.bincount(y)[1] / len(y)

    # Compute the minimum PR curve
    recall = np.linspace(0, 1, 100)
    p_min = []
    for r in recall:
        mp = (pi * r) / (pi * r + (1 - pi))
        p_min.append(mp)
    unachievable = pd.DataFrame({"precision": p_min, "recall": recall})

    # Compute The area of the unachievable region in PR space and the minimum AUCPR
    unachievable_area = 1 + (((1 - pi) * log(1 - pi)) / pi)

    # Print a text on the unachievable region
    text_y = unachievable.precision[80] / 2
    text = (
        alt.Chart({"values": [{"x": 0.8, "y": text_y}]})
        .mark_text(text="Unachievable Area")
        .encode(x="x:Q", y="y:Q")
    )

    # Plot the two graphs
    base_pr = (
        alt.Chart(pr)
        .encode(
            y=alt.Y("precision", axis=alt.Axis(format="%", title="Precision")),
            x=alt.X("recall", axis=alt.Axis(format="%", title="Recall")),
        )
        .mark_line(color=color)
        .properties(
            title={
                "text": "PR-Curve",
                "subtitle": f"AUCPR = {auc_:.2f} - AUCPRN = {(auc_ - unachievable_area)/(1 - unachievable_area):.2f}",
            }
        )
    )

    unachievable_region = (
        alt.Chart(unachievable)
        .encode(
            y=alt.Y("precision", axis=alt.Axis(format="%")),
            x=alt.X("recall", axis=alt.Axis(format="%")),
        )
        .mark_area(opacity=0.1)
    )

    full = base_pr + unachievable_region + text

    return full
