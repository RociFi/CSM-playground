import pandas as pd
import numpy as np
from scipy import stats

from xgboost import XGBClassifier

loaded_model = XGBClassifier()
loaded_model.load_model('models/9.0.5/xgboost_model_v_9.0.5.bin')

df = pd.read_csv("example.csv")

borrows_limit = 1
deflated_min = 0.0018688429
deflated_max = 0.9998634
t_stat_percentile = 0.9
max_score = 10
min_score = 1


def convert_prob_to_score(y_pred: np.ndarray) -> np.ndarray:
    """Convert model probability to the credit score from 0 to 10."""
    deflated_value_scaled = y_pred - deflated_min
    deflated_value_scaled = deflated_value_scaled / \
        (deflated_max - deflated_min)  # type: ignore
    deflated_value_scaled = deflated_value_scaled * (max_score - min_score)
    deflated_value_scaled = (max_score - min_score) - deflated_value_scaled
    return deflated_value_scaled + min_score


def count_borrow(y_pred: np.ndarray, count_borrows: np.ndarray, borrows_limit: int, t_stat_percentile: float) -> np.ndarray:
    """Reduces the final gauge according to the number of loans."""
    borrows_limit_mask = count_borrows > borrows_limit
    stdev = np.divide(y_pred * (1 - y_pred), (count_borrows - 1),
                      out=np.zeros_like(y_pred), where=count_borrows != 1)
    stdev = np.where(borrows_limit_mask, stdev, 0)

    # Find the t_stat_percentile quantile in t-distribution:
    quantile: np.ndarray = stats.t.ppf(t_stat_percentile, count_borrows - 1)

    y_pred = y_pred - quantile * stdev
    y_pred = np.where(borrows_limit_mask, y_pred, 0)
    return np.where(y_pred > 0, y_pred, 0)


def transform_predictions(y_pred, count_borrows):
    y_pred = count_borrow(predicted_proba, count_borrows,
                          borrows_limit, t_stat_percentile)
    scores = convert_prob_to_score(y_pred)
    scores[count_borrows <= borrows_limit] = max_score
    return scores.round()


features = loaded_model.predictor.features

predicted_proba = loaded_model.predict_proba(features)[:, 1]

count_borrows = df["count_borrow"].values
predictions_from_loaded_bin = transform_predictions(
    predicted_proba, count_borrows)
print(predictions_from_loaded_bin)
