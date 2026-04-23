# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a **Random Forest Classifier**, with class imbalance handled via **SMOTE** and model explainability powered by **SHAP**.

---

## 📌 Problem Statement

Credit card fraud is a major financial threat. This project builds a binary classifier to identify fraudulent transactions from a highly imbalanced real-world dataset, where fraud cases represent less than 1% of all transactions.

---

## 📂 Dataset

- **File:** `creditcard.csv` ❌ *Not included in this repository*
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 (V1–V28 are PCA-transformed, plus `Time` and `Amount`)
- **Target:** `Class` → 0 = Legitimate, 1 = Fraud


⚠️ **Note:** The dataset is not included in this repository due to its size.  
Please download it manually from the Kaggle link above and place it in the project root directory:
---

## 🧠 Approach

| Step | Details |
|------|---------|
| **EDA** | Checked shape, class distribution, missing values |
| **Split** | 80/20 train-test split with stratification |
| **Scaling** | StandardScaler applied (fit on train, transform on test) |
| **Balancing** | SMOTE applied on training data only (after split) |
| **Model** | Random Forest (100 estimators, n_jobs=-1) |
| **Evaluation** | Classification report, ROC-AUC, PR-AUC |
| **Explainability** | SHAP TreeExplainer — global & local explanations |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.97+ |
| PR-AUC | ~0.85+ |
| Precision (Fraud) | High |
| Recall (Fraud) | High |



---

## 🖼️ Visualizations

All plots are saved in the `images/` folder:

| Plot | Description |
|------|-------------|
| `confusion_matrix.png` | True vs predicted labels |
| `roc_curve.png` | ROC curve with AUC score |
| `shap_bar.png` | Top 15 features by SHAP importance (global) |
| `shap_beeswarm.png` | SHAP beeswarm — feature impact direction & magnitude |
| `shap_local.png` | SHAP force plot for a single fraud prediction (local) |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data | Pandas, NumPy |
| ML | Scikit-learn, imbalanced-learn (SMOTE) |
| Model | Random Forest Classifier |
| Explainability | SHAP |
| Visualization | Matplotlib |

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/SeemaShivanappanavar/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib shap
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook credit_card.ipynb
   ```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── credit_card.ipynb       # Main notebook
├── creditcard.csv          # Dataset (add the downloaded dataset here)
├── README.md               # Project documentation
└── images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── shap_bar.png
    ├── shap_beeswarm.png
    └── shap_local.png
```

---

## 👩‍💻 Author

**Seema Shivanappanavar**  
B.E. ECE — KLE Technological University, Hubli  
[GitHub](https://github.com/SeemaShivanappanavar)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
