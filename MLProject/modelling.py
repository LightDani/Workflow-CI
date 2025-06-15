import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.sklearn import log_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def save_metric_summary(summary):
    with open("metrics_summary.json", "w") as f:
        json.dump(summary, f)
    mlflow.log_artifact("metrics_summary.json")
    os.remove("metrics_summary.json")


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap="coolwarm", fmt="d")
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("conf_matrix.png")
    mlflow.log_artifact("conf_matrix.png")
    plt.close()
    # os.remove("conf_matrix.png")


def main(train_path, test_path, max_iter):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)

    X_train = data_train.iloc[:, :-1]
    y_train = data_train["Personality"]
    X_test = data_test.iloc[:, :-1]
    y_test = data_test["Personality"]

    model = LogisticRegression(max_iter=max_iter)

    with mlflow.start_run():
        mlflow.set_tag("model_name", "LogisticRegression")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_train, y_pred),
            "precision": precision_score(y_train, y_pred, average="macro"),
            "recall": recall_score(y_train, y_pred, average="macro"),
            "f1": f1_score(y_train, y_pred, average="macro"),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, average="macro"),
            "test_recall": recall_score(y_test, y_test_pred, average="macro"),
            "test_f1": f1_score(y_test, y_test_pred, average="macro"),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        save_metric_summary(metrics)
        mlflow.log_params(model.get_params())

        signature = infer_signature(X_train, y_pred)
        log_model(model, "model", signature=signature, input_example=X_test.iloc[:5])

        save_confusion_matrix(y_test, y_test_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path"
    )  # , type=str, default="personality_preprocessing/train.csv")
    parser.add_argument(
        "--test_path"
    )  # , type=str, default="personality_preprocessing/test.csv")
    parser.add_argument("--max_iter")  # , type=int, default=1000)
    args = parser.parse_args()

    main(args.train_path, args.test_path, args.max_iter)
