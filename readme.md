# Michelin Restaurant Recommender

**AIPI 540 — Duke University**  
_Evan Moh_

---

## Overview

This project builds a **personalized Michelin restaurant recommender for the US** using a **hybrid approach** that combines both **collaborative filtering** (SVD matrix factorization) and **content-based filtering** (feature engineering).

---

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Data Sources](#data-sources)
- [Pipeline Steps](#pipeline-steps)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Ethics Statement](#ethics-statement)

---

## File Structure```

AIPI540_Final/<br>
├── data/<br>
│ ├── raw/<br>
│ │ ├── Michelin_List.xlsx<br>
│ │ ├── MichelinColabFiltering.xlsx<br>
│ │ └── Michelin Restaurant Recommender.xlsx<br>
│ └── output/<br>
│ ├── cleaned_colab_feedback.csv<br>
│ ├── cleaned_survey_feedback.csv<br>
│ ├── user_features.csv<br>
│ ├── restaurant_features.csv<br>
│ ├── training_hybrid.csv<br>
│ ├── training_hybrid_wcf.csv<br>
│ └── unmatched_for_review.csv<br>
├── models/<br>
│ ├── best_ml_model.joblib<br>
│ ├── best_nn.pt<br>
│ ├── imputer.joblib<br>
│ ├── scaler.joblib<br>
│ ├── svd_model.joblib<br>
│ ├── svd_mappings.json<br>
│ ├── feature_cols.json<br>
│ ├── user_latent.npy<br>
│ └── rest_latent.npy<br>
├── scripts/<br>
│ ├── data_prep.py<br>
│ ├── features.py<br>
│ ├── ml.py<br>
│ ├── nn.py<br>
│ ├── naive.py<br>
│ ├── svd_train.py<br>
│ └── app.py<br>
├── requirements.txt<br>
├── setup.py<br>
├── LICENSE<br>
└── README.md<br>
---

## Data Sources

- **SurveyMonkey Survey**: Michelin/fine-dining users from Reddit
- **Reddit Posts**: Fine dining, Michelin communities
- **Michelin Website**: Star info, cuisine, locations

---

## Pipeline Steps

### 1. Data Preparation (`scripts/data_prep.py`)
- Loads and cleans data from survey, colab feedback, and Michelin list.
- Fuzzy-matches user restaurant input to official Michelin names.
- Builds user-restaurant "like/dislike" labels.

### 2. Feature Engineering (`scripts/features.py`)
- **User features**: Extracted from survey (e.g., enjoys wine, cares about service, memorable experiences, disappointment keywords).
- **Restaurant features**: Encodes price, cuisine, location, Michelin star, etc.
- Outputs final feature tables.

### 3. Hybrid Matrix & Collaborative Filtering (`scripts/svd_train.py`)
- Builds user-restaurant interaction matrix from like/dislike feedback.
- Runs SVD (matrix factorization) to extract collaborative filtering signals (latent factors).
- Saves user and restaurant latent factors and mapping dictionaries.

### 4. Modeling
- **Naive Baseline (`scripts/naive.py`)**: Always recommends most popular restaurants.
- **Classical ML (`scripts/ml.py`)**: Trains Random Forest, Logistic Regression, and XGBoost using hybrid user/restaurant features (+ SVD latent score). Picks the best via validation ROC-AUC.
- **Deep Learning (`scripts/nn.py`)**: PyTorch neural net trained on the same feature set. Hyperparameter tuning and early stopping are included.

### 5. Evaluation
- **Test ROC-AUC**: Ability to distinguish liked vs not liked.
- **Classification metrics**: Precision, recall, f1-score.
- **Ranking metrics**: Precision@5, Recall@5, NDCG@5 — measures quality of the *top* recommendations for each user.

---

## How to Run
**install requirements**
```bash
pip install -r requirements.txt
``` 
***Run setup.py***


## Chosen Approach
- Classical ML (XGBoost) gives the most balanced and accurate recommendations overall - so XGBoost was chosen.
- Neural Net achieves very high recall but at the cost of slightly lower precision and NDCG, which can sometimes mean more false positives or less ranking quality.


##  Ethics Statement
- **User Privacy**: All survey/feedback data anonymized and handled securely.
- **Responsible Use**: Intended for supportive, non-commercial purposes only.
- **Data Usage**: Collected with user knowledge and used solely for academic research.

## Notes
- All modeling code is in scripts/
- Data outputs go to data/output/
- Model artifacts saved to models/

