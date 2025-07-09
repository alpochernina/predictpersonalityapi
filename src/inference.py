import numpy as np
import pandas as pd
import joblib
import pickle
from src.data_preprocessing import preprocess_test_data


class PersonalityPredictor:
    def __init__(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        with open(encoder_path, "rb") as f:
            self.label_encoders, self.expected_cols, self.drop_cols, self.object_cols = pickle.load(f)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return preprocess_test_data(df, self.drop_cols, self.object_cols, self.label_encoders)

    def predict(self, df: pd.DataFrame):
        df = self.preprocess(df)
        predicted_personality = self.model.predict(df).tolist()[0]
        if predicted_personality == 0:
            return "Extrovert"
        else:
            return "Introvert"