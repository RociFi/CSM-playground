import pandas as pd
from xgboost import XGBClassifier
from transformations import transform_predictions

loaded_model = XGBClassifier()
loaded_model.load_model('models/9.0.6/xgboost_model_v_9.0.6.bin')

df = pd.read_csv("example.csv")

features = ['total_borrow', 'count_borrow', 'avg_borrow_amount', 'std_borrow_amount', 'borrow_amount_cv', 'total_repay', 'count_repay', 'avg_repay_amount', 'std_repay_amount', 'repay_amount_cv', 'total_deposit', 'count_deposit', 'avg_deposit_amount', 'std_deposit_amount', 'deposit_amount_cv', 'total_redeem', 'count_redeem', 'avg_redeem_amount', 'std_redeem_amount', 'redeem_amount_cv', 'total_liquidation', 'count_liquidation', 'avg_liquidation_amount',
            'std_liquidation_amount', 'liquidation_amount_cv', 'days_since_first_borrow', 'net_outstanding', 'int_paid', 'net_deposits', 'count_repays_to_count_borrows', 'avg_repay_to_avg_borrow', 'net_outstanding_to_total_borrowed', 'net_outstanding_to_total_repaid', 'count_redeems_to_count_deposits', 'total_redeemed_to_total_deposits', 'avg_redeem_to_avg_deposit', 'net_deposits_to_total_deposits', 'net_deposits_to_total_redeemed', 'avg_liquidation_to_avg_borrow']

# Get not normalized score from the model

predicted_proba = loaded_model.predict_proba(df[features])[:, 1]

count_borrows = df["count_borrow"].values

# Normalize scores

predictions_from_loaded_bin = transform_predictions(
    predicted_proba, count_borrows, predicted_proba)

print(predictions_from_loaded_bin)
