# 🏦 Loan Default Prediction & Credit Risk Modeling

An end-to-end Machine Learning project to assess credit risk and predict loan defaults using the **LendingClub** dataset. This project focuses on handling severe **class imbalance**, building a leak-free **Scikit-Learn Pipeline**, and aligning model evaluation metrics with actual financial trade-offs.

---

## 📌 Business Overview & Problem Statement
In retail banking, non-performing loans (defaults) represent a significant financial loss compared to the opportunity cost of rejecting a viable borrower.
* **Target:** `not.fully.paid` (Binary Classification: `0` = Fully Paid, `1` = Default).
* **Data Challenge:** Severe class imbalance — **84%** fully paid vs. **16%** defaulted loans.
* **Objective:** Design a model sensitive enough to detect high-risk applicants while minimizing unmanaged defaults.

---

## 🛠️ Pipeline Architecture
To ensure data integrity and avoid **data leakage**, all preprocessing steps were encapsulated inside a unified Scikit-Learn `ColumnTransformer` and `Pipeline`:
* **Stratified Split:** Used `train_test_split(stratify=y)` to maintain the 84:16 class ratio across training and validation sets.
* **Categorical Handling:** Automated categorical column detection followed by `OneHotEncoder(handle_unknown='ignore')`.
* **Numerical Features:** Passed through without alteration (`passthrough`), with zero missing values detected.

---

## 📊 The "Accuracy Paradox" & Model Evolution

A standard baseline model was misled by class imbalance, achieving an apparently high **84% accuracy** while failing to catch almost all defaulting borrowers.

To address this, we transitioned to a weighted **XGBoost Classifier** with `scale_pos_weight=5.25`, explicitly penalizing misclassifications of the minority (default) class.

### Confusion Matrix Comparison

| Metric / Outcome | Baseline Random Forest | Weighted XGBoost |
| :--- | :---: | :---: |
| **Accuracy** | ~84% (Misleading) | ~67% |
| **Captured Defaults (Recall on Class 1)** | **1.3%** (Only 4 of 307 caught) | **48.5%** (149 of 307 caught) |
| **Escaped Defaults (False Negatives)** | 303 high-risk borrowers | **158** high-risk borrowers |
| **Financial Impact** | Massive direct loss from missed defaults | Drastic loss reduction through proactive flagging |

---

## 🔍 Key Feature Drivers (Interpretability)
Using XGBoost's feature importance analysis, the top predictive factors driving loan defaults were identified:

1. **`credit.policy`:** Whether the applicant met the underwriting guidelines was the single most decisive factor.
2. **`int.rate`:** Higher interest rates created heavier payment burdens, escalating default likelihood.
3. **`purpose_small_business`:** Small business loans exhibited a notably higher failure and default rate compared to personal/consolidation loans.

---

## 💻 Tech Stack
* **Language:** Python
* **Data Analysis & Viz:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/hasteetabatabaei/Loan-Default-Prediction-Credit-Risk-Modeling.git
