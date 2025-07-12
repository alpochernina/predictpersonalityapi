## Personality Prediction API – Introvert vs Extrovert

This project was created as part of the [Kaggle Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/overview), using the [Extrovert vs Introvert Behavior Dataset](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data).

The final model is deployed via aFastAPIserver, allowing users to submit data and receive predictions through a REST API.

## Features

* EDA: Exploratory Data Analysis.
* Preprocessing: Data cleaning, encoding (for both train/test).
* Model: Trained using XGBoost classifier with optimized hyperparameters.
* Evaluation: Model validation using a confusion matrix and classification report.
* API: FastAPI server with an endpoint to classify new data.

## API Usage

Run the server:

```bash
python server.py
```

Example request:

```json
POST /predict
{
    "Time_spent_Alone": 5.0,
    "Stage_fear": "Yes",
    "Going_outside": 4.0,
    "Social_event_attendance": 5.0,
    "Drained_after_socializing": "Yes",
    "Friends_circle_size": 4.0,
    "Post_frequency": 3.0
}
```

Response:

```json
{
  "prediction": "Introvert"
}
```

## Installation

```bash
pip install -r requirements.txt
```

### Model File Not Included

The trained model file (`.pkl`) is not included in this repository.

If you want to train the model yourself:

1. Make sure the dataset is placed in the `data/raw` directory.
2. Train the model using: `python train.py`

<pre class="o
