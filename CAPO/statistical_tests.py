import math

import numpy as np
from scipy.special import rel_entr
from scipy.stats import chi2, mannwhitneyu, ttest_rel

# TODO: generally be careful with what kind of scores (0/1, continuous) are assumed
# and specify this in the docstrings or include checks for this in the functions


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


def mann_whitney_u_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Uses a Mann-Whitney U test to test if candidate A's accuracy is significantly
    higher than candidate B's accuracy within a confidence interval of 1-\alpha.
    Assumptions:
    - No parametric assumptions


    Parameters:
        scores_a (np.ndarray): Array of accuracy scores for candidate A.
        scores_b (np.ndarray): Array of accuracy scores for candidate B.
        alpha (float): Significance level (default 0.05 for 95% confidence).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """

    _, p_value = mannwhitneyu(scores_a, scores_b, alternative="greater")

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

    Assumptions:
    - Scores are bounded between min_val and max_val.

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

    n = min(len(scores_a), len(scores_b))  # TODO: should be same length -> rather check for that

    R = max_val - min_val
    epsilon_a = math.sqrt((R**2 * math.log(2 / alpha)) / (2 * n))
    epsilon_b = math.sqrt((R**2 * math.log(2 / alpha)) / (2 * n))

    result = (mean_a - epsilon_a) > (mean_b + epsilon_b)

    return result


def chernoff_bound_test_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Uses Chernoff bound with KL divergence to test if candidate A's accuracy
    is significantly higher than candidate B's accuracy.

    Candidate A is significantly better than B if:
        (mean_a - delta_a) > (mean_b + delta_b)

    Parameters:
        scores_a (np.ndarray): Array of scores for candidate A.
        scores_b (np.ndarray): Array of scores for candidate B.
        alpha (float): Significance level (default 0.05 for 95% confidence).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """
    mean_a = scores_a.mean()
    mean_b = scores_b.mean()
    n = min(len(scores_a), len(scores_b))
    target = math.log(2 / alpha) / n  # Target value for KL divergence

    def find_delta(mean: float, sign: int) -> float:
        """Finds the delta such that n * KL(mean ± delta, mean) = target."""
        low, high = 0.0, 1.0 - mean if sign > 0 else mean  # Search range
        for _ in range(50):  # Binary search
            mid = (low + high) / 2
            kl_val = sum(rel_entr([mean + sign * mid, 1 - mean - sign * mid], [mean, 1 - mean]))
            if kl_val < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    delta_a = find_delta(mean_a, -1)  # Lower bound for A
    delta_b = find_delta(mean_b, +1)  # Upper bound for B

    return (mean_a - delta_a) > (mean_b + delta_b)


def mcnemar_test_diff(
    scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05, correction: bool = True
) -> bool:
    """
    Perform McNemar's test to determine whether candidate A (with predictions y_pred1)
    is significantly better than candidate B (with predictions y_pred2).

    The test is based on the following 2x2 contingency table:

                      Candidate B
                   Correct   Incorrect
    Candidate A  -------------------------
    Correct       |    n_00   |    n_01 (b)
    Incorrect     |    n_10   |    n_11 (c)

    Only the discordant pairs (b and c) are used in the test statistic.
    The statistic is computed as:

        chi2_stat = ((|b - c| - 1)**2) / (b + c)  [with continuity correction]
        chi2_stat = ((b - c)**2) / (b + c)           [without continuity correction]

    Under the null hypothesis, chi2_stat follows a chi-squared distribution with 1 degree of freedom.

    Candidate A is declared significantly better if:
        - The test is significant (p < alpha)
        - b > c (i.e. more instances where candidate A is correct and candidate B is wrong)

    Assumptions:
    - The samples are paired.
    - The samples are binary (0 or 1).
    - Classifcation task is binary (2 classes).

    Parameters:
        scores_a (np.ndarray): Array of accuracies for candidate A.
        scores_b (np.ndarray): Array of accuracies for candidate B.
        alpha (float): Significance level (default 0.05).
        correction (bool): Whether to apply continuity correction (default True).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """
    # Count instances where candidate A is correct and B is wrong, and vice versa.
    b = np.sum((scores_a == 1) & (scores_b == 0))
    c = np.sum((scores_a == 0) & (scores_b == 1))

    # If there are no discordant pairs, there's no evidence to claim superiority.
    if (b + c) == 0:
        return False

    # Calculate the McNemar chi-squared statistic.
    if correction:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2_stat = (b - c) ** 2 / (b + c)

    # Compute the p-value from the chi-squared distribution with 1 degree of freedom.
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    # Candidate A is considered significantly better only if the test is significant and b > c.
    return (p_value < alpha) and (b > c)
