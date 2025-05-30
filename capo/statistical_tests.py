"""
Implementation of statistical significance tests used in the racing algorithm.
Contains paired t-test functionality to compare prompt performance and determine statistical significance between candidates.
"""
import numpy as np
from scipy.stats import ttest_rel


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Uses a paired t-test to test if candidate A's accuracy is significantly
    higher than candidate B's accuracy within a confidence interval of 1-\alpha.
    Assumptions:
    - The samples are paired.
    - The differences between the pairs are normally distributed (-> n > 30).

    Parameters:
        scores_a (np.ndarray): Array of accuracy scores for candidate A.
        scores_b (np.ndarray): Array of accuracy scores for candidate B.
        alpha (float): Significance level (default 0.05 for 95% confidence).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """

    _, p_value = ttest_rel(scores_a, scores_b, alternative="greater")

    result = p_value < alpha

    return result
