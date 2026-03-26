# 🏥 Predictive Analytics in Medical Diagnosis

A machine learning project that predicts diseases based on patient symptoms using four classification algorithms — **Naive Bayes**, **Random Forest**, **SVM (Support Vector Machine)**, and **Decision Tree (J48)**.

> **Project by:** Mukku Takur | Sk. Hasan | M. Harsha | T. Rishi Vardhan  
> **Institution:** Aditya University, Department of AIML  
> **Guide:** Mrs. S. Pravallika, M.Tech.

---

## 📁 Project Structure

```
disease_prediction/
│
├── data/
│   └── dps_dataset.csv          # Disease-symptoms dataset (99 records)
│
├── models/
│   ├── naive_bayes.py           # Naive Bayes classifier
│   ├── random_forest.py         # Random Forest classifier
│   ├── svm_classifier.py        # SVM (Support Vector Machine)
│   └── decision_tree.py         # Decision Tree (J48 equivalent)
│
├── utils/
│   ├── preprocessor.py          # Data loading & preprocessing
│   └── evaluator.py             # Model evaluation & metrics
│
├── results/                     # Auto-generated result plots & reports
│
├── main.py                      # Run all models & compare results
├── predict.py                   # Predict disease from symptoms (CLI tool)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run All Models

```bash
python main.py
```

### 4. Predict a Disease from Symptoms

```bash
python predict.py --symptoms "fever,cough,fatigue,headache"
```

---

## 📊 Algorithm Results (10-Fold Cross Validation)

| Algorithm       | Accuracy | Kappa  |
|-----------------|----------|--------|
| Naive Bayes     | **80.8%**| 0.8000 |
| Random Forest   | 77.56%   | 0.7662 |
| SVM (SMO)       | 76.64%   | 0.7567 |
| Decision Tree   | 56.84%   | 0.5504 |

✅ **Best Model: Naive Bayes** with 80.8% accuracy

---

## 🔬 Dataset

- **Records:** 99 disease entries
- **Columns:** `disease`, `symptoms`, `cures`, `doctor`, `risk level`
- **Diseases Covered:** Flu, Diabetes, Cancer, Heart Attack, Stroke, Pneumonia, Malaria, and 90+ more

---

## 📦 Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

---

## 📜 License

This project is for academic purposes. © 2026 Aditya University AIML Department.
