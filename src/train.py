"""
train.py

Train and evaluate Linear Regression models, comparing two versions:
with Room vs. without Room.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from data_preprocessing import load_data, split_data, build_preprocessor, get_feature_names


def train_model(X_train, y_train, preprocessor):
    """
    Build a pipeline consisting of the preprocessor and a
    Linear Regression model, then fit it on the training set.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test, label=""):
    """
    Evaluate the model using MAE, RMSE, and R² on the test set.
    """
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"--- {label} ---")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print()

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}


if __name__ == '__main__':
    # Fixed paths (no automatic path construction)
    data_path = 'D:/File Code/house-price-prediction/data/house_prices_clean.csv'
    model_dir = 'D:/File Code/house-price-prediction/models'
    model_path = model_dir + '/linear_regression_best.pkl'

    # Create the models directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Version 1: With Room ---
    preprocessor_with_room = build_preprocessor(include_room=True)
    pipeline_with_room = train_model(X_train, y_train, preprocessor_with_room)
    result_with_room = evaluate_model(
        pipeline_with_room,
        X_test,
        y_test,
        label="With Room"
    )

    # --- Version 2: Without Room ---
    preprocessor_no_room = build_preprocessor(include_room=False)
    pipeline_no_room = train_model(X_train, y_train, preprocessor_no_room)
    result_no_room = evaluate_model(
        pipeline_no_room,
        X_test,
        y_test,
        label="Without Room"
    )

    # --- Compare the two models and save the better one ---
    if result_with_room['r2'] >= result_no_room['r2']:
        best_pipeline = pipeline_with_room
        print(">> Selected model: WITH Room (higher or equal R²)")
    else:
        best_pipeline = pipeline_no_room
        print(">> Selected model: WITHOUT Room (higher R²)")

    joblib.dump(best_pipeline, model_path)
    print(f"Model saved to: {model_path}")