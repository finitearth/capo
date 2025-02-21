import math
import numpy as np
from scipy.stats import ttest_rel



def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Uses a paired t-test to test if candidate A's accuracy is significantly
    higher than candidate B's accuracy.

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
        

def hoeffdings_inequality_test_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> bool:
    """
    Uses Hoeffding's inequality to test if candidate A's accuracy is significantly
    higher than candidate B's accuracy when they have different numbers of evaluations.

    For a candidate with n evaluations and observed average score, Hoeffding's inequality
    gives a confidence bound:
        epsilon = sqrt((R^2 * log(2/delta)) / (2*n))
    where R = max_val - min_val.

    Candidate A is considered significantly better than candidate B if:
        (score_a - epsilon_a) > (score_b + epsilon_b)

    Parameters:
        scores_a (np.ndarray): Array of scores for candidate A.
        scores_b (np.ndarray): Array of scores for candidate B.
        n (int): Number of independent evaluations.
        alpha (float): Significance level (default 0.05 for 95% confidence).
        min_val (float): Minimum possible score (default 0.0).
        max_val (float): Maximum possible score (default 1.0).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """
    mean_a = scores_a.mean()
    mean_b = scores_b.mean()

    n = min(len(scores_a), len(scores_b))

    R = max_val - min_val
    epsilon_a = math.sqrt((R**2 * math.log(2 / alpha)) / (2 * n))
    epsilon_b = math.sqrt((R**2 * math.log(2 / alpha)) / (2 * n))

    result = (mean_a - epsilon_a) > (mean_b + epsilon_b)

    return result
