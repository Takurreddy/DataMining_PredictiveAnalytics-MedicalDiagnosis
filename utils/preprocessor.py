"""
utils/preprocessor.py
---------------------
Loads the disease-symptoms dataset and transforms it into a
binary feature matrix suitable for machine learning classifiers.

Each symptom becomes a binary column (1 = present, 0 = absent).
The target label is the disease name.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress sklearn warnings about high class-to-sample ratio
warnings.filterwarnings('ignore', message='.*number of unique classes.*')

# ── Path to the dataset ──────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dps_dataset.csv')

# ── Dataset note ─────────────────────────────────────────────────────────────
# dps_dataset.csv is a disease-reference file (99 records, 84 diseases).
# The full training dataset used in WEKA had 2500 records with 50 binary
# symptom attributes. Replace data/dps_dataset.csv with the larger dataset
# to reproduce the WEKA accuracy results from the project report.


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_symptom_matrix(df: pd.DataFrame):
    """
    Parse the comma-separated 'symptoms' column and create a binary
    feature matrix where each unique symptom is a column.

    Returns
    -------
    X : pd.DataFrame  — binary symptom feature matrix
    y : pd.Series     — disease labels (string)
    all_symptoms : list — sorted list of all unique symptoms
    """
    # ── Collect all unique symptoms ──────────────────────────────────────────
    all_symptoms = set()
    symptom_lists = []

    for raw in df['symptoms']:
        symptoms = [s.strip().lower() for s in str(raw).split(',')]
        symptom_lists.append(symptoms)
        all_symptoms.update(symptoms)

    all_symptoms = sorted(all_symptoms)

    # ── Build binary matrix ──────────────────────────────────────────────────
    rows = []
    for symptoms in symptom_lists:
        row = {sym: 1 if sym in symptoms else 0 for sym in all_symptoms}
        rows.append(row)

    X = pd.DataFrame(rows, columns=all_symptoms)
    y = df['disease'].str.strip().str.lower()

    return X, y, all_symptoms


def encode_labels(y: pd.Series):
    """Encode string disease labels to integers. Returns encoded y and the encoder."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def get_preprocessed_data(test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    le                                : fitted LabelEncoder
    all_symptoms                      : list of symptom feature names
    """
    df = load_raw_data()
    X, y, all_symptoms = build_symptom_matrix(df)
    y_encoded, le = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_encoded, test_size=test_size,
        random_state=random_state, stratify=None
    )

    return X_train, X_test, y_train, y_test, le, all_symptoms


def symptoms_to_vector(user_symptoms: list, all_symptoms: list) -> np.ndarray:
    """
    Convert a user-supplied list of symptom strings into a binary
    feature vector aligned with all_symptoms.

    Parameters
    ----------
    user_symptoms : list of str  (e.g. ['fever', 'cough', 'fatigue'])
    all_symptoms  : list of str  (full feature list from training data)

    Returns
    -------
    vector : np.ndarray shape (1, len(all_symptoms))
    """
    user_symptoms = [s.strip().lower() for s in user_symptoms]
    vector = np.array([[1 if sym in user_symptoms else 0 for sym in all_symptoms]])
    return vector


if __name__ == '__main__':
    df = load_raw_data()
    print(f"Dataset shape : {df.shape}")
    print(f"Columns       : {list(df.columns)}")

    X, y, all_symptoms = build_symptom_matrix(df)
    print(f"Feature matrix: {X.shape}  ({len(all_symptoms)} unique symptoms)")
    print(f"Disease classes: {y.nunique()}")
    print(f"\nSample features (first 10): {all_symptoms[:10]}")
    print(f"\nClass distribution:\n{y.value_counts()}")
