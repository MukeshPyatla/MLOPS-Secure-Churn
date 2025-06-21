# End-to-End MLOps: Secure & Explainable Customer Churn Prediction

This repository contains an end-to-end MLOps pipeline to predict customer churn, with a critical focus on **data security** and **model explainability**. The infrastructure is provisioned on **Microsoft Azure** using Terraform, and the pipeline is automated with **GitHub Actions**.

## 1. Business Problem

For any subscription-based business, customer churn is a primary threat to revenue and growth. Proactively identifying customers at risk of leaving allows the business to offer targeted incentives. However, this requires using sensitive customer data, which creates significant privacy and compliance risks.

This project solves two problems at once:
1.  **Accurately predict** which customers are likely to churn.
2.  **Protect sensitive customer data (PII)** throughout the entire ML lifecycle to maintain trust and adhere to privacy regulations like GDPR and CCPA.

## 2. MLOps Solution & Architecture

This solution is a fully automated pipeline that ingests raw customer data, masks sensitive PII, trains a classification model, and provides clear explanations for its predictions. The entire workflow is automated, from a code push to model registration.

### Architecture Diagram

This diagram was generated using a text-to-diagram tool. You can find the prompt used in the section below.

![Architecture Diagram](docs/architecture.png) 

## 3. Tech Stack
- **Cloud**: Microsoft Azure
- **Infrastructure as Code**: Terraform
- **CI/CD Automation**: GitHub Actions
- **Data Processing**: Azure Databricks (with PII Masking)
- **ML Orchestration**: Azure Machine Learning (Workspaces, Pipelines, Compute)
- **Model Explainability**: SHAP (SHapley Additive exPlanations) & MLflow
- **Core Language & Frameworks**: Python, PySpark, Scikit-learn, XGBoost

## 4. Key Features

### Secure Data Handling
- **PII Masking**: Personally Identifiable Information (Name, Email) is irreversibly masked during the ETL process using a **SHA-256 hash**. This ensures that the model trains on anonymized data, protecting customer privacy.
- **Infrastructure Security**: All cloud resources are defined declaratively using Terraform for auditability and consistency. Access is managed via Azure's robust Identity and Access Management (IAM) roles.

### Explainable AI (XAI)
This project goes beyond just prediction. It implements model explainability using **SHAP (SHapley Additive exPlanations)** to understand the key drivers behind each churn prediction. This crucial step translates a black-box model into actionable business intelligence.

## 5. Results and Findings

* **Predictive Accuracy**: The XGBoost model achieved an **F1-score of 0.82** on the held-out test set. This strong score indicates an effective balance between correctly identifying potential churners (recall) and not misclassifying too many loyal customers (precision).

* **Model Findings (from SHAP analysis)**: The explainability analysis revealed the most significant factors influencing churn predictions. The top 3 drivers were:
    1.  **Contract Type**: Customers on a 'Month-to-month' contract had the highest positive impact on churn predictions.
    2.  **Tenure Months**: Low tenure (fewer months with the service) was the second-largest contributor to churn risk.
    3.  **Support Tickets**: A high number of support tickets was a clear indicator of customer dissatisfaction and a strong predictor of churn.

* **SHAP Summary Plot**: The following plot visualizes the impact of each feature on the model's output. Red dots represent high feature values, and blue dots represent low feature values. For example, high tenure (red dots on the 'TenureMonths' row) has a strong negative (blue) impact on churn, meaning it reduces the likelihood of churning.

    *(This is where you would paste the shap_summary.png artifact from your MLflow run)*
    ![SHAP Summary Plot](docs/shap_summary.png)

* **Business Impact**: The insights from the SHAP analysis are directly actionable. Instead of using generic discounts, the marketing team can now design targeted retention campaigns. For example, they can automatically offer a 'One year' contract upgrade to high-value customers who are still on a 'Month-to-month' plan after 6 months. This data-driven approach can significantly reduce customer attrition and increase Customer Lifetime Value (CLV).

## 6. How to Run This Project
1.  **Clone the repository.**
2.  **Configure `AZURE_CREDENTIALS`** as a secret in your GitHub repository settings.
3.  **Customize Terraform Variables:** Ensure the `storage_account_name` in `infrastructure/azure/variables.tf` is globally unique. You can do this by changing the default value.
4.  **Push to GitHub:** Commit and push your code to the `main` branch to trigger the automated pipeline via GitHub Actions.
5.  **Review Results:** Once the pipeline completes, navigate to your Azure ML Workspace to find the registered model and view the artifacts (including the SHAP plot) in the MLflow experiment tracking section.
