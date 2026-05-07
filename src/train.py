import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Cấu hình MLflow đúng chuẩn bài lab
DB_PATH = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(DB_PATH)

# Dùng experiment đã tạo từ Grid Search
experiment_name = "WineQualityLab"
mlflow.set_experiment(experiment_name)

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình một lần duy nhất với bộ tham số tối ưu.
    """

    # Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    with mlflow.start_run(run_name="Production_Run"):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Lưu file cho Bước 2 CI/CD
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        return acc


if __name__ == "__main__":
    if not os.path.exists("params.yaml"):
        print("Lỗi: Không tìm thấy file params.yaml.")
        exit(1)
        
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        
    print(f"Đang huấn luyện lần cuối với bộ tham số tốt nhất: {params}")
    train(params)
