# Tabular Classification Hackathon — Readmission Prediction

This repository contains my end-to-end tabular ML pipeline for a hackathon-style classification task.  
The goal is to predict **hospital readmission within 30 days** using clinical and administrative features.

The project is implemented as a single Python script that includes:
- Fast EDA checks (missing values, duplicates, target distribution)
- Feature engineering tailored to common hospital datasets (ICD-9 grouping, “measured” flags)
- A robust preprocessing pipeline (imputation + scaling + one-hot encoding)
- Baseline comparison (DummyClassifier) and a stronger model (Logistic Regression)
- Evaluation (Accuracy, Macro F1, Confusion Matrix + optional ROC/PR plots)
- A helper utility to **fill the target column for the final submission CSV** using the trained pipeline

---

## Why these choices?

### 1) Data cleaning
- **Missing values:** The dataset uses `"?"` as a missing marker. I convert it to `NaN` so pandas/sklearn can treat it correctly.
- **Binary target:** Original target has 3 values (`"<30"`, `">30"`, `"NO"`). I map it to:
  - `1` = readmitted in **<30 days**
  - `0` = otherwise

This matches the hackathon objective: detect the “early readmission” class.

---

### 2) Feature engineering decisions

#### Diagnosis codes → medical “chapters”
Raw diagnosis codes (`diag_1`, `diag_2`, `diag_3`) are high-cardinality and noisy.  
I compress ICD-9 codes into broad medical groups (e.g., Diabetes, Circulatory, Respiratory, etc.) and then **drop the raw diagnosis columns**.

This reduces sparsity and helps the model generalize better, especially in a hackathon environment.

#### “Measured” flags
Some clinical attributes (`weight`, `A1Cresult`, `max_glu_serum`) are missing very often.  
Instead of using their raw values directly, I create boolean indicators like:
- `weight_measured`
- `A1Cresult_measured`
- `max_glu_serum_measured`

This preserves signal (“was it measured?”) even when the value itself is sparse.

#### ID-like numeric columns treated as categorical
Columns such as:
- `admission_type_id`
- `discharge_disposition_id`
- `admission_source_id`

look numeric but represent **categories**. I cast them to string so they are one-hot encoded rather than treated as ordinal numeric values.

---

### 3) Feature selection / dropping columns
I drop:
- **Unique identifiers**: `encounter_id`, `patient_nbr` (do not generalize; risk leakage)
- Columns that are not useful or are extremely sparse in raw form (handled via “measured” flags)
- Other dataset-specific columns that are not informative for learning in this setup (e.g., `examide`, `citoglipton`)

---

### 4) Preprocessing pipeline (sklearn)
I use a `ColumnTransformer` with two pipelines:

**Numeric pipeline**
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

**Categorical pipeline**
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)`

Notes:
- `handle_unknown="ignore"` prevents failures when unseen categories appear in validation/final data.
- `min_frequency=0.01` reduces extreme sparsity by grouping rare categories.

---

### 5) Models

#### Baseline: DummyClassifier
A `DummyClassifier(strategy="most_frequent")` provides a baseline that reflects the “naive guess” in an imbalanced dataset.

#### Main model: Logistic Regression
I use Logistic Regression because it is:
- Strong and fast for tabular baselines
- Interpretable (coefficients)
- Reliable under time constraints

Key configuration:
- `class_weight="balanced"` to address class imbalance (minority class = readmitted <30 days)
- `solver="lbfgs"`, `max_iter=2000`
- `C=0.5` (mild regularization)

---

## Handling class imbalance (important lesson)
Initially, Logistic Regression **without** class weighting produced high Accuracy but extremely poor Recall for the minority class (near zero).  
This showed that **Accuracy alone is not sufficient** for imbalanced medical classification.  
I therefore evaluated using **Macro F1** and enabled `class_weight="balanced"`.

---

## Evaluation
The script reports:
- Accuracy
- Macro F1
- Confusion Matrix
- Classification Report (Precision / Recall / F1 per class)
- Stratified 10-fold cross-validation (Macro F1)

It also includes optional plots:
- Confusion Matrix
- ROC curve (if `predict_proba` exists)
- Precision-Recall curve (if `predict_proba` exists)

---

## Filling the final submission file
A key helper function:
`fill_final_target_from_model(...)`

It:
1. Loads the final CSV (without labels)
2. Applies the **same feature engineering**
3. Aligns columns to the training feature set (adds missing columns, removes extra columns, preserves order)
4. Predicts using the trained pipeline
5. Writes a new CSV with:
   - `readmitted` predictions
   - optional probability column `readmitted_proba_yes`

---

## How to run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run
```bash
python HackathonTabularClassification.py
```

---

## Reproducibility
- Train/test split uses `random_state=42`
- Stratified 10-fold CV uses `random_state=42`

---