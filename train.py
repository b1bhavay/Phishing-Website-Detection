
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare phishing detection models"
    )

    parser.add_argument(
        "--data",
        default="data/Phishing_Websites_Data.csv"
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db"
    )
    parser.add_argument(
        "--experiment-name",
        default="Phishing-Website-Detection"
    )

    args = parser.parse_args()

    # 1. Load dataset
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()

    target_col = next(
        (col for col in df.columns if col.lower() == "result"),
        None
    )

    if target_col is None:
        raise ValueError("Target column 'Result' not found.")

    # 2. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Feature selection (same as project notebook)
    features_to_remove = [
        "favicon",
        "double_slash_redirecting"
    ]

    missing = [f for f in features_to_remove if f not in X.columns]
    if missing:
        raise ValueError(f"Expected feature columns not found: {missing}")

    X = X.drop(columns=features_to_remove)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 5. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42
        )
    }

    # 7. MLflow setup
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    best_f1 = -1
    best_model = None
    best_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, pos_label=1
                ),
                "recall": recall_score(
                    y_test, predictions, pos_label=1
                ),
                "f1_score": f1_score(
                    y_test, predictions, pos_label=1
                )
            }

                       mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("features_count", X_train.shape[1])

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                serialization_format="cloudpickle"
            )
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("features_count", X_train.shape[1])
            mlflow.log_metrics(metrics)
           mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    serialization_format="cloudpickle"
)

            print(f"\n{name}")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_name = name

    # 8. Save best model and preprocessing files
    os.makedirs("model", exist_ok=True)

    joblib.dump(best_model, "model/model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump(list(X.columns), "model/features.joblib")

    print("\nBest model:", best_name)
    print(f"Best F1-score: {best_f1:.4f}")
    print("Model and preprocessing files saved successfully.")


if __name__ == "__main__":
    main()
