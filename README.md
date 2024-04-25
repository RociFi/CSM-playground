# Quick start:

```
$ git clone https://github.com/RociFi/CSM-playground
$ cd CSM-playground
$ python predict.py
```

This example script uses [example.csv](example.csv) as an input. To use the model in a production environment, replace input data with the realtime data for the given wallet.

# Intro

RociFi credit risk score is a metric which reflects a wallet's likelihood of defaulting on their debts by analyzing DeFi transaction history and behavior. RociFi’s credit score scale is 1–10 where 1 is the lowest credit risk (best score) and 10 is the highest credit risk (worst score).

Under the hood, credit risk score is a ML model which employs a combination of cross-validation and hyperparameter optimization. For more information, please refer to [this article](https://betterprogramming.pub/ai-powered-credit-scoring-on-the-blockchain-building-an-ml-model-for-undercollateralized-defi-376707fe91fd).

# Data used by the model

Wallet’s transaction (borrow/lend/etc) history for the following lending protocols:

- Aave
- Compound
- Cream
- RociFi
- Venus
- MakerDAO
- GMX
- Radiant

Transaction history on following chains:

- Ethereum (full transaction history)
- Arbitrum (protocol specific history)
- Fantom (protocol specific history)
- Polygon (protocol specific history)
- Optimism (protocol specific history)
- BSC (protocol specific history)
- Avalanche (protocol specific history)
- Historical token prices for conversion to USDT on transaction time

[example.csv](example.csv) contains example dataset.

List of features

There are 5 groups of features:
Borrow
Repay
Deposit
Redeem
Liquidation
Derived

Borrow and repay are self-explanatory, deposit means deposit of collateral, redeem is collateral withdrawal. Derived is a feature of a feature.

All amounts should be denominated in USD.

- total_borrow: total borrowed amount
- count_borrow: borrow count
- avg_borrow_amount: average borrowed amount
- std_borrow_amount: borrowed amount, standard deviation
- borrow_amount_cv: borrowed amount, variation
- total_repay: total repaid amount
- count_repay: repayment count
- avg_repay_amount: average repaid amount
- std_repay_amount: repaid amount, standard deviation
- repay_amount_cv: repaid amount, variation
- total_deposit: total deposit amount
- count_deposit: deposit count
- avg_deposit_amount: average deposit amount
- std_deposit_amount: deposit amount, standard deviation
- deposit_amount_cv: deposit amount, variation
- total_redeem: total redeemed amount
- count_redeem: redeeming count
- avg_redeem_amount: average redeemed amount
- std_redeem_amount: redeemed amount, standard deviation
- redeem_amount_cv: redeemed amount, variation
- total_liquidation: total liquidated amount
- count_liquidation: liquidations count
- avg_liquidation_amount: average liquidated amount
- std_liquidation_amount': liquidated amount, standard deviation
- liquidation_amount_cv: liquidated amount, variation
- days_since_first_borrow': days since first borrow transaction
- net_outstanding: total debt
- int_paid: same as total_repay
- net_deposits: same as total_deposit
- count_repays_to_count_borrows: count_repays / count_borrows
- avg_repay_to_avg_borrow: avg_repay / avg_borrow
- net_outstanding_to_total_borrowed: net_outstanding / total_borrowed',
- net_outstanding_to_total_repaid: net_outstanding / total_repaid',
- count_redeems_to_count_deposits: count_redeems / count_deposits',
- total_redeemed_to_total_deposits: total_redeemed / total_deposits',
- avg_redeem_to_avg_deposit: avg_redeem / avg_deposit',
- net_deposits_to_total_deposits: net_deposits / total_deposits',
- net_deposits_to_total_redeemed: net_deposits / total_redeemed',
- avg_liquidation_to_avg_borrow: avg_liquidation / avg_borrow'
