"""
models/decision_tree.py
-----------------------
Decision Tree classifier for disease prediction.

A Decision Tree splits the data recursively on the symptom feature that
provides the most information gain, building a tree of if-then rules.

Equivalent to the WEKA configuration:
  J48 Decision Tree — C 0.25 (confidence factor), M 2 (min instances per leaf)
  Produced 372 leaves, tree size 743

This model achieved 56.84% accuracy in the project (lowest among four).
The large tree (743 nodes) suggests overfitting on the small dataset.

scikit-learn mapping to WEKA J48:
  criterion='entropy'           → matches C4.5 information gain splitting
  min_samples_leaf=2            → matches WEKA's -M 2
  ccp_alpha                     → post-pruning, approximates WEKA's -C 0.25
"""

import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.pipeline import Pipeline


def build_decision_tree_model(
    criterion: str = 'entropy',
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 2,
    ccp_alpha: float = 0.01,
    random_state: int = 42
):
    """
    Build and return a Decision Tree pipeline (J48 equivalent).

    Parameters
    ----------
    criterion         : 'entropy' (C4.5 / J48) or 'gini' (CART)
    max_depth         : int or None — maximum tree depth (None = grow fully)
    min_samples_split : int         — min samples to split a node
    min_samples_leaf  : int         — min samples in leaf (WEKA -M 2)
    ccp_alpha         : float       — pruning complexity parameter (≈ WEKA -C 0.25)
    random_state      : int         — random seed

    Returns
    -------
    sklearn Pipeline with DecisionTreeClassifier
    """
    model = Pipeline([
        ('classifier', DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        ))
    ])
    return model


def print_tree_stats(model):
    """Print tree structure statistics after fitting."""
    dt = model.named_steps['classifier']
    n_leaves = dt.get_n_leaves()
    depth    = dt.get_depth()
    n_nodes  = dt.tree_.node_count

    print(f"\n  Decision Tree Statistics:")
    print(f"    Number of leaves : {n_leaves}")
    print(f"    Tree depth       : {depth}")
    print(f"    Total nodes      : {n_nodes}")


def print_tree_rules(model, feature_names: list, max_depth: int = 3):
    """Print the top levels of the decision tree as human-readable rules."""
    dt = model.named_steps['classifier']
    rules = export_text(dt, feature_names=feature_names, max_depth=max_depth)
    print(f"\n  Decision Tree Rules (first {max_depth} levels):")
    print(rules)


def save_tree_plot(model, feature_names: list, class_names: list,
                   output_path: str = None, max_depth: int = 4):
    """Save a visual plot of the decision tree."""
    if output_path is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'decision_tree_plot.png')

    dt = model.named_steps['classifier']

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax
    )
    ax.set_title('Decision Tree (J48 Equivalent) — Disease Prediction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Tree plot saved → {output_path}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from utils.preprocessor import get_preprocessed_data
    from utils.evaluator import evaluate_model, cross_validate_model
    import numpy as np

    print("=" * 60)
    print("  DECISION TREE (J48) — Disease Prediction")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le, all_symptoms = get_preprocessed_data()
    model = build_decision_tree_model()

    # ── 10-Fold Cross Validation ─────────────────────────────────────────────
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_results = cross_validate_model(model, X_all, y_all, n_folds=10)
    print(f"\n  10-Fold CV Accuracy: {cv_results['mean_accuracy']*100:.2f}% "
          f"(± {cv_results['std_accuracy']*100:.2f}%)")

    # ── Train/Test Evaluation ────────────────────────────────────────────────
    model.fit(X_train, y_train)
    print_tree_stats(model)
    print_tree_rules(model, all_symptoms, max_depth=3)

    evaluate_model(model, X_train, X_test, y_train, y_test, le, 'Decision Tree (J48)')

    # ── Tree Visualisation ───────────────────────────────────────────────────
    class_names = list(le.classes_)
    save_tree_plot(model, all_symptoms, class_names)
