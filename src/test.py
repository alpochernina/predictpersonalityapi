import numpy as np
import pandas as pd
import joblib

xgb_model = joblib.load("model/xgb_model.pkl")

test_data = pd.read_csv("data/processed/test_processed.csv").sort_values(by="id")
ids = test_data["id"].to_numpy()
X_test = test_data.drop(columns=["id"])

xgb_pred_test = xgb_model.predict(X_test)

personality = np.where(xgb_pred_test == 0, 'Extrovert', 'Introvert')

df = pd.DataFrame({
    'id': ids,
    'Personality': personality
})

df.to_csv('output/personality_output.csv', index=False)

print(test_data.dtypes)