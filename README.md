# End-to-End MLOps: Secure & Explainable Customer Churn Prediction

This repository contains an end-to-end MLOps pipeline to predict customer churn, with a critical focus on **data security** and **model explainability**. The infrastructure is provisioned on **Microsoft Azure** using Terraform, and the pipeline is automated with **GitHub Actions**.

## 1. Business Problem

For subscription-based businesses, customer churn is a major threat to revenue. Proactively identifying customers at risk of leaving allows businesses to offer targeted incentives, but this requires using sensitive customer data. This project solves two problems at once:
1.  Accurately predict which customers are likely to churn.
2.  Protect sensitive customer data (PII) throughout the entire ML lifecycle to maintain trust and compliance.

## 2. MLOps Solution & Architecture

This solution is a fully automated pipeline that:
- Ingests raw customer data containing PII.
- **Masks sensitive data** during the ETL process.
- Trains a machine learning model to predict churn.
- **Generates explainability insights** to show *why* a prediction was made.
- Is fully automated via CI/CD.

### Architecture Diagram
**(Create a new diagram for this workflow and embed it here)**
![Architecture Diagram](docs/architecture.png)

## 3. Tech Stack
- **Cloud**: Microsoft Azure
- **Infrastructure as Code**: Terraform
- **CI/CD Automation**: GitHub Actions
- **Data Processing**: Azure Databricks (with PII Masking)
- **ML Orchestration**: Azure Machine Learning (Workspaces, Pipelines, Compute)
- **Model Explainability**: SHAP (SHapley Additive exPlanations)
- **Core Language**: Python (PySpark, Scikit-learn, XGBoost)

## 4. Key Features

### Secure Data Handling
- **PII Masking**: Personally Identifiable Information (Name, Email) is irreversibly masked during the ETL process using SHA-256 hashing to protect customer privacy.
- **Infrastructure Security**: Resources are defined in code for auditability, and access is managed via Azure's identity and access management.

### Explainable AI (XAI)
This project goes beyond just prediction. It implements model explainability using **SHAP** to understand the key drivers behind each churn prediction. This empowers business teams to create effective, data-driven retention strategies.

**(Embed your SHAP summary plot here after you run the pipeline)**
![SHAP Summary Plot](docs/shap_summary.png)

## 5. How to Run This Project
**(Update these instructions as needed)**
1.  **Clone the repository.**
2.  **Configure `AZURE_CREDENTIALS`** as a secret in your GitHub repository.
3.  **Customize Terraform Variables:** Ensure `storage_account_name` in `infrastructure/azure/variables.tf` is globally unique.
4.  **Push to GitHub** to trigger the pipeline.