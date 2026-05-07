import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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

        model_type = params.get("model_type", "random_forest")
        model_kwargs = {k: v for k, v in params.items() if k != "model_type"}

        if model_type == "xgboost":
            if "min_samples_split" in model_kwargs: del model_kwargs["min_samples_split"]
            model = XGBClassifier(**model_kwargs, random_state=42, use_label_encoder=False, eval_metric="mlogloss")
        else:
            if "learning_rate" in model_kwargs: del model_kwargs["learning_rate"]
            model = RandomForestClassifier(**model_kwargs, random_state=42)

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 5: Cảnh báo lệch lạc dữ liệu
        class_dist = y_train.value_counts(normalize=True).to_dict()
        dist_dict = {int(k): float(v) for k, v in class_dist.items()}
        print("\n--- Phân phối dữ liệu huấn luyện ---")
        for cls, ratio in dist_dict.items():
            print(f"Lớp {cls}: {ratio*100:.2f}%")
            if ratio < 0.10:
                print(f"⚠️ CẢNH BÁO: Lớp {cls} chiếm quá ít dữ liệu ({ratio*100:.2f}% < 10%). Nguy cơ Data Drift!")

        # Bonus 3: Báo cáo hiệu suất
        cm = confusion_matrix(y_eval, preds)
        report = classification_report(y_eval, preds)
        
        report_text = f"=== BÁO CÁO HIỆU SUẤT ===\n\n1. Confusion Matrix:\n{cm}\n\n2. Precision, Recall, F1:\n{report}"
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        # Lưu metrics.json (thêm data_distribution từ Bonus 5)
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_score": f1,
                "data_distribution": dist_dict
            }, f)

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
