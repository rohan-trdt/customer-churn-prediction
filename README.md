# Customer Churn Prediction

A machine learning web app that predicts whether a telecom customer is likely to churn, based on their account details and usage patterns.

## Problem Statement
Customer churn is a major cost driver for subscription-based businesses. This project predicts churn risk using historical customer data, helping identify at-risk customers before they leave.

## Dataset
Telco Customer Churn dataset (Kaggle) — 7,043 customer records with 20 features including tenure, contract type, monthly charges, and service subscriptions.

## Approach
- Cleaned and preprocessed data (handled missing values, encoded categorical features)
- Trained and compared Logistic Regression and Random Forest models
- Random Forest achieved ~79% accuracy
- Identified key churn drivers using feature importance analysis: **Total Charges, Monthly Charges, Tenure, and Contract Type** were the strongest predictors
- Built an interactive web app using Streamlit for real-time churn prediction

## Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (web app)
- Joblib (model persistence)

## Live Demo
https://customer-churn-prediction-r.streamlit.app/

## How to Run Locally

pip install -r requirements.txt
streamlit run app.py


## Key Insight
Customers on month-to-month contracts with high monthly charges and low tenure are at the highest risk of churning — suggesting retention efforts should prioritize new customers without long-term contracts.