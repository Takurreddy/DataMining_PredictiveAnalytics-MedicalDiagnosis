"""
models/naive_bayes.py
---------------------
Naive Bayes classifier for disease prediction.

Naive Bayes is a probabilistic classifier based on Bayes' theorem with
a strong (naive) assumption of feature independence. It is well-suited
for binary feature data and handles multi-class problems naturally.

Why it works well here:
  - Symptom features are binary (present / absent) → BernoulliNB is ideal
  - Even with many symptoms, independence assumption gives reliable results
  - Fast training and inference; robust with limited data

This model achieved the BEST accuracy in the project: 80.8%
"""

from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


def build_naive_bayes_model(alpha: float = 1.0):
    """
    Build and return a Bernoulli Naive Bayes pipeline.

    Parameters
    ----------
    alpha : float
        Laplace/Lidstone smoothing parameter (default 1.0).
        Higher values increase smoothing and reduce overfitting.

    Returns
    -------
    sklearn Pipeline with BernoulliNB
    """
    model = Pipeline([
        ('classifier', BernoulliNB(alpha=alpha))
    ])
    return model


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from utils.preprocessor import get_preprocessed_data
    from utils.evaluator import evaluate_model, cross_validate_model

    print("=" * 60)
    print("  NAIVE BAYES — Disease Prediction")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le, all_symptoms = get_preprocessed_data()
    model = build_naive_bayes_model()

    # ── 10-Fold Cross Validation ─────────────────────────────────────────────
    import numpy as np
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_results = cross_validate_model(model, X_all, y_all, n_folds=10)
    print(f"\n  10-Fold CV Accuracy: {cv_results['mean_accuracy']*100:.2f}% "
          f"(± {cv_results['std_accuracy']*100:.2f}%)")

    # ── Train/Test Evaluation ────────────────────────────────────────────────
    evaluate_model(model, X_train, X_test, y_train, y_test, le, 'Naive Bayes')
