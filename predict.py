"""
predict.py
----------
Command-line tool to predict a disease from a list of symptoms.

Uses the best-performing model (Naive Bayes, 80.8% accuracy) by default.
Also shows the top-3 likely diseases with confidence percentages.

Usage:
    python predict.py --symptoms "fever,cough,fatigue,headache"
    python predict.py --symptoms "chest pain,shortness of breath,nausea" --model rf
    python predict.py --list-symptoms

Arguments:
    --symptoms   : Comma-separated list of symptoms (required unless --list-symptoms)
    --model      : Model to use: nb (Naive Bayes), rf (Random Forest),
                                 svm (SVM), dt (Decision Tree)  [default: nb]
    --top        : Number of top predictions to show             [default: 3]
    --list-symptoms : Print all available symptoms and exit
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import (
    load_raw_data, build_symptom_matrix,
    encode_labels, symptoms_to_vector
)
from models.naive_bayes    import build_naive_bayes_model
from models.random_forest  import build_random_forest_model
from models.svm_classifier import build_svm_model
from models.decision_tree  import build_decision_tree_model


MODEL_MAP = {
    'nb':  ('Naive Bayes',     build_naive_bayes_model),
    'rf':  ('Random Forest',   build_random_forest_model),
    'svm': ('SVM (RBF)',       build_svm_model),
    'dt':  ('Decision Tree',   build_decision_tree_model),
}


def load_extra_info(disease_name: str) -> dict:
    """Look up doctor and risk level from the raw dataset for a given disease."""
    df = load_raw_data()
    df['disease'] = df['disease'].str.strip().str.lower()
    match = df[df['disease'] == disease_name]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {
        'cures':      row.get('cures', 'N/A'),
        'doctor':     row.get('doctor', 'N/A'),
        'risk_level': row.get('risk level', 'N/A'),
    }


def predict_disease(symptoms: list, model_key: str = 'nb', top_n: int = 3):
    """
    Train the chosen model and predict the disease from given symptoms.

    Parameters
    ----------
    symptoms  : list of symptom strings
    model_key : 'nb' | 'rf' | 'svm' | 'dt'
    top_n     : number of top predictions to show
    """
    # ── Prepare data ──────────────────────────────────────────────────────────
    df = load_raw_data()
    X, y, all_symptoms = build_symptom_matrix(df)
    y_encoded, le = encode_labels(y)

    # ── Validate symptoms ──────────────────────────────────────────────────────
    clean_symptoms = [s.strip().lower() for s in symptoms]
    unknown = [s for s in clean_symptoms if s not in all_symptoms]
    matched = [s for s in clean_symptoms if s in all_symptoms]

    if unknown:
        print(f"\n  ⚠  Unknown symptom(s) ignored: {', '.join(unknown)}")
    if not matched:
        print("\n  ✗  No recognised symptoms provided.")
        print("     Run:  python predict.py --list-symptoms")
        sys.exit(1)

    # ── Train model ───────────────────────────────────────────────────────────
    model_name, model_fn = MODEL_MAP[model_key]
    model = model_fn()
    model.fit(X.values, y_encoded)

    # ── Predict ───────────────────────────────────────────────────────────────
    vector = symptoms_to_vector(matched, all_symptoms)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector)[0]
        top_indices = np.argsort(proba)[::-1][:top_n]
        top_diseases = [(le.classes_[i], proba[i]) for i in top_indices]
    else:
        pred = model.predict(vector)[0]
        top_diseases = [(le.classes_[pred], 1.0)]

    # ── Display Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  DISEASE PREDICTION REPORT")
    print(f"  Model : {model_name}")
    print("=" * 60)
    print(f"\n  Symptoms entered  : {', '.join(matched)}")
    print(f"\n  Top {top_n} Predicted Diseases:\n")

    for rank, (disease, conf) in enumerate(top_diseases, 1):
        info = load_extra_info(disease)
        marker = "  ◀ MOST LIKELY" if rank == 1 else ""
        print(f"  {rank}. {disease.title():<35} Confidence: {conf*100:.1f}%{marker}")
        if info:
            print(f"     Doctor      : {info.get('doctor', 'N/A')}")
            print(f"     Risk Level  : {info.get('risk_level', 'N/A')}")
            print(f"     Cures       : {info.get('cures', 'N/A')[:80]}...")
        print()

    print("=" * 60)
    print("  ⚠  This is an educational tool. Always consult a doctor.\n")


def list_all_symptoms():
    """Print all available symptom keywords."""
    df = load_raw_data()
    _, _, all_symptoms = build_symptom_matrix(df)
    print(f"\n  {len(all_symptoms)} available symptoms:\n")
    for i, sym in enumerate(sorted(all_symptoms), 1):
        print(f"  {i:>3}. {sym}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Disease Prediction CLI — Predictive Analytics Project'
    )
    parser.add_argument(
        '--symptoms', type=str,
        help='Comma-separated list of symptoms e.g. "fever,cough,fatigue"'
    )
    parser.add_argument(
        '--model', type=str, default='nb',
        choices=list(MODEL_MAP.keys()),
        help='Model to use: nb=Naive Bayes, rf=Random Forest, svm=SVM, dt=Decision Tree'
    )
    parser.add_argument(
        '--top', type=int, default=3,
        help='Number of top disease predictions to show (default: 3)'
    )
    parser.add_argument(
        '--list-symptoms', action='store_true',
        help='Print all available symptom keywords and exit'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list_symptoms:
        list_all_symptoms()
        sys.exit(0)

    if not args.symptoms:
        print("\n  ✗  Please provide symptoms using --symptoms \"fever,cough,...\"")
        print("     Or use --list-symptoms to see all available keywords.\n")
        sys.exit(1)

    symptom_list = [s.strip() for s in args.symptoms.split(',')]
    predict_disease(symptom_list, model_key=args.model, top_n=args.top)
