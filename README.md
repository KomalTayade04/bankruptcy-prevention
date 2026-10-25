# 📉 Bankruptcy Prevention: Classification Project

Predicts whether a company is likely to go **bankrupt** or **not**, based on six qualitative risk factors, so that lenders and investors can reduce their losses and companies can act early.

🔗 **Live app:** https://bankruptcy-prevention-xsxvgadtn8yih6qeway5kg.streamlit.app/

## Problem Statement
Lending to or investing in a company that later goes bankrupt causes heavy financial loss. This project builds a classification model that flags high-risk companies early.

## Dataset
- 250 companies, 6 input features and 1 target (`class`: bankruptcy / non-bankruptcy)
- No missing values
- Each feature is rated **0 (Low), 0.5 (Medium) or 1 (High)**

| Feature | Meaning of a high value |
|---|---|
| industrial_risk | Riskier industry |
| management_risk | Riskier management |
| financial_flexibility | Stronger ability to handle financial problems |
| credibility | More trustworthy |
| competitiveness | Stronger against competitors |
| operating_risk | Riskier day-to-day operations |

## Approach
1. **EDA:** class balance (57% non-bankruptcy, 43% bankruptcy), feature distributions by class, correlation heatmap
2. **Duplicates:** 147 of 250 rows are duplicates, since each feature has only 3 possible values (103 unique profiles). They were kept, but identical profiles are never split between train and test, to avoid data leakage.
3. **Model comparison:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting and XGBoost, using stratified group 5-fold cross-validation
4. **Tuning:** GridSearchCV on Logistic Regression, optimising **recall** for the bankruptcy class (missing a company that fails costs more than a false alarm)
5. **Deployment:** interactive Streamlit app

## Key Findings
- **Competitiveness, credibility and financial flexibility** are the strongest protective factors against bankruptcy.
- **Industrial, operating and management risk** increase bankruptcy risk, but less strongly.
- Every company with competitiveness = 0 in the dataset went bankrupt.

## Results
Final model: **Logistic Regression** (C = 0.1), evaluated on a held-out test set of 50 unseen companies.

| Metric | Score |
|---|---|
| Accuracy | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| ROC-AUC | 1.00 |

Random Forest scored lower on the same test set (accuracy 0.92, recall 0.81), so Logistic Regression was chosen. It also generalises better and is easy to explain.

> ⚠️ **Limitations:** the dataset is small, and the classes are almost perfectly separable, so the perfect score should not be expected on real-world data. The model should be validated on more data before real use.

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit

## Project Structure
```
├── app.py                      # Streamlit app
├── models/bankruptcy_model.pkl # Trained model
├── 01_eda_and_modeling.ipynb   # EDA and model building
└── requirements.txt            # Dependencies
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
