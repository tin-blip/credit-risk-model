# Credit Risk Probability Model for Alternative Data

## Project Overview

This project builds an end-to-end credit risk prediction system for Bati Bank using alternative transaction data from an eCommerce platform. The system predicts customer credit risk probability to support buy-now-pay-later services.

The project includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Proxy Target Variable Creation using RFM Analysis
- Machine Learning Model Training
- MLflow Experiment Tracking
- FastAPI Deployment
- Docker Containerization
- CI/CD Automation

---

## Project Structure

**credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md**
----------------

# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability

The Basel II Accord emphasizes proper risk measurement, transparency, and regulatory compliance in financial systems. Because credit risk models influence lending decisions, financial institutions must ensure that models are interpretable, well-documented, and auditable.

An interpretable model allows stakeholders, regulators, and risk managers to understand how predictions are generated. This improves trust, accountability, and compliance with regulatory requirements. Proper documentation also helps institutions monitor model performance, identify biases, and validate that the model behaves consistently over time.

In this project, interpretability is important because the model will support credit approval decisions that directly affect customers and financial risk exposure.

---

## 2. Importance of Proxy Variables

The dataset does not contain a direct default label indicating whether a customer failed to repay a loan. Because supervised machine learning models require labeled target variables, a proxy variable must be created to represent customer risk behavior.

This project uses customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) analysis to identify potentially high-risk customers. Customers with low engagement, infrequent transactions, and low monetary activity may represent higher credit risk.

However, proxy-based prediction introduces business risks because the proxy does not represent actual loan default behavior. Incorrect assumptions during proxy creation may lead to biased predictions, unfair customer treatment, or inaccurate credit decisions. Therefore, the proxy target must be carefully designed, justified, and monitored.

---

## 3. Trade-offs Between Logistic Regression and Gradient Boosting

Logistic Regression is a simple and interpretable model commonly used in traditional credit scoring. When combined with Weight of Evidence (WoE) encoding, it provides clear explanations of how features influence risk predictions. This transparency makes it attractive in regulated financial environments.

However, Logistic Regression may struggle to capture complex nonlinear relationships in the data, potentially reducing predictive performance.

Gradient Boosting models, such as XGBoost or LightGBM, often achieve higher predictive accuracy because they can learn complex patterns and feature interactions. These models may improve risk prediction performance significantly.

The trade-off is that Gradient Boosting models are less interpretable and more difficult to explain to regulators and business stakeholders. In regulated financial contexts, institutions must balance predictive performance with transparency, fairness, maintainability, and compliance requirements.

---

## Author

10 Academy KAIM Week 4 Challenge
