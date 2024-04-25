import numpy as np
from scipy import stats

"""Don't change these unless you know what you're doing"""

borrows_limit = 1
deflated_min = 0.0018688429
deflated_max = 0.9998634
t_stat_percentile = 0.9
max_score = 10
min_score = 1

"""Normalize model probability to the range 0 to 10."""


def convert_prob_to_score(y_pred: np.ndarray) -> np.ndarray:
    deflated_value_scaled = y_pred - deflated_min
    deflated_value_scaled = deflated_value_scaled / \
        (deflated_max - deflated_min)  # type: ignore
    deflated_value_scaled = deflated_value_scaled * (max_score - min_score)
    deflated_value_scaled = (max_score - min_score) - deflated_value_scaled
    return deflated_value_scaled + min_score


"""Reduces the final gauge according to the number of loans."""


def count_borrow(y_pred: np.ndarray, count_borrows: np.ndarray, borrows_limit: int, t_stat_percentile: float) -> np.ndarray:
    borrows_limit_mask = count_borrows > borrows_limit
    stdev = np.divide(y_pred * (1 - y_pred), (count_borrows - 1),
                      out=np.zeros_like(y_pred), where=count_borrows != 1)
    stdev = np.where(borrows_limit_mask, stdev, 0)

    # Find the t_stat_percentile quantile in t-distribution:
    quantile: np.ndarray = stats.t.ppf(t_stat_percentile, count_borrows - 1)

    y_pred = y_pred - quantile * stdev
    y_pred = np.where(borrows_limit_mask, y_pred, 0)
    return np.where(y_pred > 0, y_pred, 0)


""" Normalize score with borrows data"""


def transform_predictions(y_pred, count_borrows, predicted_proba):
    y_pred = count_borrow(predicted_proba, count_borrows,
                          borrows_limit, t_stat_percentile)
    scores = convert_prob_to_score(y_pred)
    scores[count_borrows <= borrows_limit] = max_score
    return scores.round()
