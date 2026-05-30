import os
import pandas as pd
import numpy as np


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load raw transaction data.
    """

    df = pd.read_csv(file_path)

    return df

def create_aggregate_features(df):
    """
    Create customer-level aggregate features.
    """

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            average_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
        )
        .reset_index()
    )

    return agg_df

def extract_time_features(df):
    """
    Extract datetime-based features.
    """

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["transaction_day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["transaction_month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["transaction_year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df

def build_preprocessing_pipeline(df):
    """
    Build preprocessing pipeline.
    """

    numerical_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col != "is_high_risk"
    ]

    categorical_cols = [
        col
        for col in df.select_dtypes(include="object").columns
        if col not in [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
        ]
    ]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_cols,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_cols,
            ),
        ]
    )

    return preprocessor

def process_data(df):
    """
    Full feature engineering workflow.
    """

    df = extract_time_features(df)

    rfm = create_rfm_features(df)

    rfm = create_high_risk_labels(rfm)

    agg_features = create_aggregate_features(df)

    df = df.merge(
        agg_features,
        on="CustomerId",
        how="left",
    )

    df = df.merge(
        rfm[["CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"]],
        on="CustomerId",
        how="left",
    )

    preprocessor = build_preprocessing_pipeline(df)

    processed_data = preprocessor.fit_transform(df)

    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
    processed_data,
    columns=feature_names,
    )

    processed_df["is_high_risk"] = df["is_high_risk"].values

    current_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.dirname(current_dir)

    processed_path = os.path.join(
        project_root,
        "data",
        "processed",
    )

    os.makedirs(processed_path, exist_ok=True)

    output_file = os.path.join(
    processed_path,
    "processed_data.csv",
    )

    processed_df.to_csv(
    output_file,
    index=False,
    )

    print(f"Processed data saved to: {output_file}")

    return processed_data, preprocessor

def create_rfm_features(df):
    """
    Create RFM metrics for customers.
    """

    snapshot_date = (
        pd.to_datetime(df["TransactionStartTime"]).max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - pd.to_datetime(x).max()
                ).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm

def create_high_risk_labels(rfm):
    """
    Cluster customers and assign high-risk labels.
    """

    scaler = MinMaxScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = (
        rfm.groupby("cluster")[
            ["Recency", "Frequency", "Monetary"]
        ]
        .mean()
    )

    print(cluster_summary)

    high_risk_cluster = (
        cluster_summary["Frequency"].idxmin()
    )

    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0,
    )

    return rfm

if __name__ == "__main__":

    file_path = "data/raw/data.csv"

    df = load_data(file_path)

    rfm = create_rfm_features(df)

    print(rfm.head())

    processed_data, preprocessor = process_data(df)

    print("Data processing completed.")

    print(processed_data.shape)