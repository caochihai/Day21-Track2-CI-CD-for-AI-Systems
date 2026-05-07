from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import uvicorn

app = FastAPI(title="Wine Quality Prediction API")

# Cấu hình các đường dẫn (Đọc từ biến môi trường nếu có)
GCS_BUCKET = os.environ.get("CLOUD_BUCKET", "wine-data-production")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    try:
        if not os.path.exists(os.path.dirname(MODEL_PATH)):
            os.makedirs(os.path.dirname(MODEL_PATH))
        
        print(f"Đang tải model từ bucket: {GCS_BUCKET}...")
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print("Tải model thành công!")
        return True
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return False

# Tải model ngay khi khởi động API
download_model()
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Kiểm tra trạng thái server và model."""
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "bucket": GCS_BUCKET
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """Nhận 12 đặc trưng và trả về kết quả dự đoán."""
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được tải lên máy chủ.")
    
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Cần cung cấp đủ 12 đặc trưng hóa học (Wine Quality).")

    # Dự đoán
    prediction = int(model.predict([req.features])[0])
    labels = {0: "thấp", 1: "trung bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "không xác định")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
