import numpy as np
import pandas as pd
import os
import subprocess
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

processed_path = "data/processed/train_processed.csv"

if not os.path.exists(processed_path):
    subprocess.run(["python", "src/data_preprocessing.py"], check=True)

data = pd.read_csv(processed_path).sort_values(by="id")

X = data.drop(columns=["id","Personality"])
y = data["Personality"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 250),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
        'gamma': trial.suggest_float('gamma', 2.4, 2.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 3.0, 3.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 2.9, 3.3), 
        'eval_metric': 'mlogloss',
        'enable_categorical': True,
        'tree_method': 'hist'
    }

    model = xgb.XGBClassifier(**params)

    y_pred = cross_val_predict(model, X, y, cv=5)

    return float(f1_score(y, y_pred, average='weighted'))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

final_params = {
    **study.best_params,
    'eval_metric': 'mlogloss',
    'enable_categorical': True,
    'tree_method': 'hist'
}

print("Best parameters found: ", final_params)

xgb_model = xgb.XGBClassifier(**final_params)
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, 'model/xgb_model.pkl')



