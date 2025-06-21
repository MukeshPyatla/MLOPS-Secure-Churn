import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sha2
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

def main(raw_data_path, processed_output_path):
    """
    Main ETL logic with PII masking and feature engineering.
    """
    spark = SparkSession.builder.appName("SecureChurnETL").getOrCreate()

    print(f"Reading raw data from {raw_data_path}")
    df = spark.read.csv(raw_data_path, header=True, inferSchema=True)

    # --- PII Masking Step ---
    print("Masking PII columns (Name, Email)...")
    df_secure = df.withColumn("HashedEmail", sha2(col("Email"), 256)).drop("Name", "Email")

    # --- Feature Engineering ---
    print("Performing feature engineering...")
    categorical_cols = ['ContractType']
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec") for c in categorical_cols]
    
    numerical_cols = ['TenureMonths', 'SupportTickets', 'MonthlyCharge', 'TotalCharges']
    assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    
    model = pipeline.fit(df_secure)
    df_features = model.transform(df_secure)

    df_final = df_features.select(col("Churn").alias("label"), col("features"), "CustomerID", "HashedEmail")

    print(f"Writing processed and secured data to {processed_output_path}")
    df_final.write.mode("overwrite").parquet(processed_output_path)
    
    print("Secure ETL job completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw input data")
    parser.add_argument("--processed_output_path", type=str, required=True, help="Path to store processed data")
    args = parser.parse_args()
    
    main(args.raw_data_path, args.processed_output_path)