import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import pickle

def preprocess_train_data(train_data: pd.DataFrame, drop_cols: list, target: list):
    train_target = train_data["Personality"]
    train_data = train_data.drop(columns=["Personality"])
    
    object_cols = train_data.select_dtypes(include='object').columns
    label_encoders = {}

    for col in object_cols:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        label_encoders[col] = le

    le_target = LabelEncoder()
    train_target = le_target.fit_transform(train_target)
    
    train_data = train_data.drop(columns=drop_cols)

    Q1 = train_data['Time_spent_Alone'].quantile(0.25)
    Q3 = train_data['Time_spent_Alone'].quantile(0.75)
    IQR = Q3 - Q1

    train_data = train_data[(train_data['Time_spent_Alone'] >= Q1 - 1.5 * IQR) & 
                (train_data['Time_spent_Alone'] <= Q3 + 1.5 * IQR)]
    
    train_data = pd.concat([train_data, pd.Series(train_target, name="Personality")], axis=1)
    
    with open("model/xgb_encoders.pkl", "wb") as f:
        pickle.dump((label_encoders, train_data.drop(columns=["id","Personality"]).columns.to_list(), drop_cols, object_cols.to_list()), f)

    return train_data, object_cols.to_list(), label_encoders, le_target

def preprocess_test_data(test_data: pd.DataFrame, drop_cols: list, object_cols: list, label_encoders: dict) -> pd.DataFrame:
    for col in object_cols:
        le = label_encoders[col]
        test_data[col] = test_data[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    return test_data.drop(columns=drop_cols)

raw_data_path = "data/raw/"
train_path = os.path.join(raw_data_path, "train.csv")
test_path = os.path.join(raw_data_path, "test.csv")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Файл не найден: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Файл не найден: {test_path}")

train_data = pd.read_csv(train_path).sort_values(by="id")
test_data = pd.read_csv(test_path).sort_values(by="id")

drop_cols = ["Going_outside"]
target = ["Personality"]

train_data, object_cols, label_encoders, le_target = preprocess_train_data(train_data, drop_cols, target)
test_data = preprocess_test_data(test_data, drop_cols, object_cols, label_encoders)

train_data.to_csv("data/processed/train_processed.csv", index=False)
test_data.to_csv("data/processed/test_processed.csv", index=False)