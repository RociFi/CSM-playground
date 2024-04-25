# CSM-playground

Credit Risk Model Playground

Credit Risk Score is a XGBoost model, returning score for the given feature vector. It's a metric shows likelihood of defaulting on the given address debts by analyzing DeFi transaction history. It's based on risk, meaning the lower the better. 1 is the lowest credit risk (best score) and 10 is the highest credit risk (worst score).

Model expects 39 features:

Core

Derived

You can find example dataset in the example.csv file.

To run the model:
$ python predict.py example.csv
