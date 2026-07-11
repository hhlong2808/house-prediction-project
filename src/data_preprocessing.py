"""
data_preprocessing.py

Data preprocessing module for the House Price Prediction project (Tehran dataset).
Target: Price_log
Features: Area_log, Room (optional), Parking, Warehouse, Elevator, Address (one-hot encoded)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


TARGET = 'Price_log'
NUMERIC_COLS_BASE = ['Area_log']          # Always used
ROOM_COL = ['Room']                       # Optional feature, enabled/disabled depending on the model version
BOOLEAN_COLS = ['Parking', 'Warehouse', 'Elevator']
CATEGORICAL_COLS = ['Address']
MIN_ADDRESS_COUNT = 20   # Addresses with fewer than this number of samples will be grouped into 'Other'

def load_data(path: str, min_count: int = MIN_ADDRESS_COUNT) -> pd.DataFrame:
    """
    Load the cleaned dataset produced during the EDA stage and group
    addresses with a small number of samples into 'Other' to avoid
    generating too many sparse one-hot encoded features.
    """
    df = pd.read_csv(path)

    # Ensure boolean columns are stored as 0/1 integers,
    # in case they are loaded as strings from the CSV file.
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(bool).astype(int)

    # Group rare addresses into 'Other'
    addr_counts = df['Address'].value_counts()
    rare_addr = addr_counts[addr_counts < min_count].index
    n_rare = len(rare_addr)
    df['Address'] = df['Address'].replace(rare_addr, 'Other')

    print(f"Grouped {n_rare} rare addresses (< {min_count} samples) into 'Other'")
    print(f"Number of remaining unique addresses: {df['Address'].nunique()}")

    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and test sets.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor(include_room: bool = True) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Parameters
    ----------
    include_room : bool
        True  -> use Area_log + Room
        False -> use only Area_log (exclude Room to avoid potential multicollinearity)
    """
    numeric_cols = NUMERIC_COLS_BASE + (ROOM_COL if include_room else [])

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('bool', 'passthrough', BOOLEAN_COLS),   # Parking/Warehouse/Elevator are already boolean, so keep them unchanged
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_COLS)
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, include_room: bool = True) -> list:
    """
    Retrieve feature names after preprocessing.
    """
    numeric_cols = NUMERIC_COLS_BASE + (ROOM_COL if include_room else [])
    cat_features = list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
    )
    return numeric_cols + BOOLEAN_COLS + cat_features


if __name__ == '__main__':
    
    data_path = 'data/house_prices_clean.csv'

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(include_room=True)
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Processed data shape:", X_train_processed.shape)
    print("Feature names:", get_feature_names(preprocessor, include_room=True))
    print(df['Address'].nunique())
    print(df['Address'].value_counts().tail(20))   # xem các khu vực có ít mẫu nhất