import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

xgb_model = joblib.load("model/xgb_model.pkl")

data = pd.read_csv("data/processed/train_processed.csv").sort_values(by="id")

X = data.drop(columns=["id","Personality"])
y = data["Personality"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

xgb_pred = xgb_model.predict(X_val)

with open("output/classification_report.txt", 'w') as f:
    f.write(str(classification_report(y_val, xgb_pred)))
    
with open("output/confusion_matrix.txt", 'w') as f:
    f.write(str(confusion_matrix(y_val, xgb_pred)))