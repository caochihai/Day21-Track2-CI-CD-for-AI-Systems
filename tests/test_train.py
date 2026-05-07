import os
import json
import numpy as np
import pandas as pd
import pytest
from src.train import train

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

def _make_temp_data(tmp_path):
    """Tạo dữ liệu giả lập cho việc test."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    
    df.iloc[:80].to_csv(train_path, index=False)
    df.iloc[80:].to_csv(eval_path, index=False)
    
    return str(train_path), str(eval_path)

def test_train_returns_accuracy(tmp_path):
    """Kiểm tra hàm train trả về giá trị accuracy hợp lệ."""
    train_path, eval_path = _make_temp_data(tmp_path)
    # Chạy thử với tham số rất nhỏ để nhanh
    acc = train({"n_estimators": 2, "max_depth": 2}, train_path, eval_path)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_metrics_file_created(tmp_path):
    """Kiểm tra file outputs/metrics.json có được tạo ra không."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train({"n_estimators": 2, "max_depth": 2}, train_path, eval_path)
    
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json", "r") as f:
        data = json.load(f)
        assert "accuracy" in data
        assert "f1_score" in data
