# Churn Prediction

A machine learning project that predicts whether a bank customer is likely to churn, deployed as a Flask web app.

## Project Structure

```
Churn_prediction/
├── templates/
│   └── index.html          Web form for entering customer details
├── app.py                  Flask application (routes + prediction logic)
├── churn_model.pkl          Trained XGBoost model
├── model_columns.pkl        Feature column order used by the model
├── Churn_Modelling.csv       Source dataset
├── Churn_Modelling.ipynb     Notebook: EDA, preprocessing, model training and tuning
├── Deployment link.txt        Link to the live deployed app
└── requirements.txt          Python dependencies
```

## Overview

The dataset contains 10,000 bank customers with attributes like credit score, geography, age, balance, and product usage, along with whether they exited (churned).

`Churn_Modelling.ipynb` covers:
- Data cleaning and feature engineering (balance-to-salary ratio, credit-score-to-age ratio)
- Baseline models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with GridSearchCV and stratified cross-validation
- Final model comparison and evaluation

The best model (tuned XGBoost) reaches **86.95%** test accuracy and is saved as `churn_model.pkl`.

## Running Locally

Install dependencies:

```
pip install -r requirements.txt
```

Start the app:

```
python app.py
```


## Tech Stack

- Python, pandas, scikit-learn, XGBoost
- Flask
- HTML/CSS (Jinja templates)
