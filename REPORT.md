# Báo Cáo Kết Quả Lab MLOps

## 1. Lựa chọn bộ siêu tham số (Hyperparameters)
- **Mô hình cuối cùng:** XGBoost (Bonus 2).
- **Bộ tham số tối ưu:** `max_depth: 6`, `n_estimators: 100`, `learning_rate: 0.1`.
- **Lý do lựa chọn:** Qua thực nghiệm ở Bước 1 (Random Forest) và sau đó nâng cấp lên XGBoost ở phần Bonus, bộ tham số này cho độ chính xác (Accuracy) ổn định trên tập Eval (> 0.63). Việc sử dụng XGBoost giúp mô hình học được các mối quan hệ phi tuyến tính tốt hơn trong dữ liệu Wine Quality.

## 2. Khó khăn gặp phải và Cách giải quyết
- **Lỗi SSH Handshake:** Trong quá trình Deploy lên VM, GitHub Actions đôi khi gặp lỗi xác thực SSH (`handshake failed`). 
  *Cách giải quyết:* Kiểm tra lại địa chỉ IP Ephemeral của VM, xác nhận đúng username `caochihai1710` và đảm bảo định dạng SSH Private Key trong GitHub Secrets chính xác.
- **Lỗi ModuleNotFoundError (pkg_resources):** Thư viện MLflow gặp lỗi không tìm thấy `pkg_resources` do phiên bản `setuptools` mới nhất đã gỡ bỏ nó.
  *Cách giải quyết:* Hạ cấp `setuptools` xuống phiên bản `69.5.1` trong file `requirements.txt` để đảm bảo tính tương thích.
- **Lỗi Database Migration (Alembic):** File `mlflow.db` bị xung đột phiên bản sau khi cài đặt lại thư viện.
  *Cách giải quyết:* Xóa file `mlflow.db` cũ để hệ thống tự động khởi tạo lại database sạch.

## 3. Các tính năng Bonus đã hoàn thành
- **Bonus 2:** Triển khai thuật toán XGBoost.
- **Bonus 3:** Tự động xuất báo cáo `report.txt` chứa Confusion Matrix.
- **Bonus 4:** Cơ chế **Rollback Protection** (Dừng deploy nếu mô hình mới tệ hơn mô hình cũ).
- **Bonus 5:** Kiểm tra và cảnh báo **Data Drift** (Lệch dữ liệu) trong quá trình huấn luyện.
