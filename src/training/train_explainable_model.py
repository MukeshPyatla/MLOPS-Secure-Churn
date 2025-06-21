import argparse
import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import matplotlib.pyplot as plt

def train_and_explain(processed_data_path, model_name):
    """
    Trains an XGBoost model and uses SHAP for explainability.
    """
    mlflow.autolog(log_models=False) # We will log the model manually

    print(f"Reading processed data from: {processed_data_path}")
    all_files = [os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path) if f.endswith('.parquet')]
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)

    X = np.array(df['features'].apply(lambda x: x.toArray()).tolist())
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    print("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, feature_names=['ContractType_One_year', 'ContractType_Two_year', 'TenureMonths', 'SupportTickets', 'MonthlyCharge', 'TotalCharges'])
    shap_summary_path = "shap_summary.png"
    plt.savefig(shap_summary_path, bbox_inches='tight')
    mlflow.log_artifact(shap_summary_path)
    os.remove(shap_summary_path)

    print(f"Registering model '{model_name}'")
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="churn-model",
        registered_model_name=model_name
    )
    
    print("Model training and explanation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str, required=True, help="Path to processed data folder.")
    parser.add_argument("--model_name", type=str, default="secure_churn_model", help="Name of the model to register.")
    args = parser.parse_args()
    
    train_and_explain(args.processed_data, args.model_name)