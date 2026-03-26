"""
models/random_forest.py
-----------------------
Random Forest classifier for disease prediction.

Random Forest is an ensemble of decision trees trained via bagging
(Bootstrap AGGregatING). Each tree is trained on a random subset of
data and features, and the final prediction is the majority vote across
all trees. This reduces overfitting and improves generalisation.

Equivalent to the WEKA configuration:
  RandomForest — 100 iterations, RandomTree base learner

This model achieved 77.56% accuracy in the project.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_random_forest_model(
    n_estimators: int = 100,
    max_features: str = 'sqrt',
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 1
):
    """
    Build and return a Random Forest pipeline.

    Parameters
    ----------
    n_estimators      : int   — number of trees (default 100, same as WEKA)
    max_features      : str   — features per split ('sqrt' mimics WEKA's -K 0)
    max_depth         : int   — max tree depth (None = unlimited)
    min_samples_split : int   — min samples to split a node (-M 1.0 in WEKA)
    min_samples_leaf  : int   — min samples in a leaf
    random_state      : int   — seed for reproducibility (-S 1 in WEKA)

    Returns
    -------
    sklearn Pipeline with RandomForestClassifier
    """
    model = Pipeline([
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1          # use all CPU cores
        ))
    ])
    return model


def get_feature_importances(model, feature_names: list, top_n: int = 15):
    """
    Extract and print top-N most important features from the trained forest.

    Parameters
    ----------
    model        : fitted Pipeline with RandomForestClassifier
    feature_names: list of symptom names
    top_n        : how many top features to display
    """
    rf = model.named_steps['classifier']
    importances = rf.feature_importances_

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  Top {top_n} Important Symptoms (Random Forest):")
    print(f"  {'Symptom':<35} {'Importance':>12}")
    print(f"  {'-'*35} {'-'*12}")
    for name, imp in pairs[:top_n]:
        print(f"  {name:<35} {imp:>12.4f}")

    return pairs


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from utils.preprocessor import get_preprocessed_data
    from utils.evaluator import evaluate_model, cross_validate_model
    import numpy as np

    print("=" * 60)
    print("  RANDOM FOREST — Disease Prediction")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le, all_symptoms = get_preprocessed_data()
    model = build_random_forest_model()

    # ── 10-Fold Cross Validation ─────────────────────────────────────────────
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_results = cross_validate_model(model, X_all, y_all, n_folds=10)
    print(f"\n  10-Fold CV Accuracy: {cv_results['mean_accuracy']*100:.2f}% "
          f"(± {cv_results['std_accuracy']*100:.2f}%)")

    # ── Train/Test Evaluation ────────────────────────────────────────────────
    results = evaluate_model(model, X_train, X_test, y_train, y_test, le, 'Random Forest')

    # ── Feature Importances ──────────────────────────────────────────────────
    get_feature_importances(model, all_symptoms)
