from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import json

# Load data from JSON file
with open('mock.json', 'r') as file:
    data = json.load(file)

data_np = np.array(list(data.values()))
# Ensure correct shape for single prediction
df = pd.DataFrame(data_np.reshape(1, -1))

print(df)

# Load the model
loaded_model = XGBClassifier()
loaded_model.load_model('models/9.0.5/xgboost_model_v_9.0.5.bin')

# Make predictions
predictions = loaded_model.predict(df)
print(predictions)
