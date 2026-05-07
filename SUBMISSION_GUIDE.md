# Hướng Dẫn Thực Hiện & Nộp Bài Lab (Chi Tiết)

Để đảm bảo bài nộp của bạn đạt điểm tối đa (80-100 điểm), hãy thực hiện theo đúng lộ trình dưới đây và lưu lại các minh chứng (ảnh chụp màn hình).

## 📂 Chuẩn Bị Thư Mục Ảnh
Trước khi bắt đầu, hãy tạo thư mục để lưu trữ ảnh cho gọn gàng:
```bash
mkdir -p submission/screenshots
```

---

## 🚀 Bước 1: Thực Nghiệm Với MLflow
1. **Mục tiêu:** Chạy 3 thí nghiệm với các bộ tham số khác nhau trong `params.yaml`.
2. **Hành động:** 
   - Chạy `python src/train.py` (Lần 1).
   - Đổi `n_estimators` hoặc `max_depth` trong `params.yaml`.
   - Chạy lại (Lần 2, Lần 3).
3. **Mở UI:** Chạy `mlflow ui` và truy cập `http://localhost:5000`.
4. **📸 CHỤP ẢNH 01:**
   - **Nội dung:** Bảng danh sách các Run trong MLflow (có đủ cột Accuracy, F1 và Params).
   - **Tên file:** `01_mlflow_experiments.png`
   - **Lưu tại:** `submission/screenshots/`

---

## 🚀 Bước 2: Quản Lý Dữ Liệu DVC & Cloud
1. **Mục tiêu:** Đẩy dữ liệu lên Cloud Storage thay vì GitHub.
2. **Hành động:** 
   - Khởi tạo DVC: `dvc init`.
   - Cấu hình Remote: `dvc remote add -d myremote gs://<bucket-name>/dvc`.
   - Đẩy dữ liệu: `dvc add data/train_phase1.csv` -> `dvc push`.
3. **📸 CHỤP ẢNH 02:**
   - **Nội dung:** Giao diện trình duyệt trang **Cloud Storage Console** (GCP/AWS/Azure) hiển thị các file bên trong thư mục `dvc/`.
   - **Tên file:** `02_cloud_storage_dvc.png`
   - **Lưu tại:** `submission/screenshots/`

---

## 🚀 Bước 3: Pipeline CI/CD (GitHub Actions)
1. **Mục tiêu:** Tự động hóa Test -> Train -> Deploy.
2. **Hành động:**
   - Cấu hình `.github/workflows/mlops.yml` và GitHub Secrets.
   - Chạy `git add .` -> `git commit -m "feat: setup pipeline"` -> `git push origin main`.
   - Đợi Pipeline chạy hoàn tất (tất cả các bước hiện màu xanh).
3. **📸 CHỤP ẢNH 03:**
   - **Nội dung:** Tab **Actions** trên GitHub, bấm vào lần chạy mới nhất để thấy đủ 4 jobs (Test, Train, Eval, Deploy) đều màu xanh.
   - **Tên file:** `03_github_actions_pipeline.png`
   - **Lưu tại:** `submission/screenshots/`

---

## 🚀 Bước 4: Kiểm Tra API (Serving)
1. **Mục tiêu:** Xác nhận mô hình đã chạy thật trên máy chủ Cloud.
2. **Hành động:**
   - Sử dụng Terminal trên máy cá nhân, gọi lệnh:
     ```bash
     curl -X POST http://<IP_CUA_VM>:8000/predict -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}"
     ```
3. **📸 CHỤP ẢNH 04:**
   - **Nội dung:** Màn hình Terminal hiển thị kết quả JSON trả về từ máy chủ (có `prediction` và `label`).
   - **Tên file:** `04_api_prediction_result.png`
   - **Lưu tại:** `submission/screenshots/`

---

## 🚀 Bước 5: Huấn Luyện Liên Tục (Bước 3 Lab)
1. **Mục tiêu:** Tự động cập nhật mô hình khi có dữ liệu mới.
2. **Hành động:** 
   - Chạy `python add_new_data.py`.
   - `dvc add` -> `dvc push`.
   - `git commit` file `.dvc` -> `git push`.
3. **Xác nhận:** Xem Pipeline tự động kích hoạt lại và cập nhật mô hình.

---

## 📝 Check-list nộp bài:
1. [ ] **GitHub URL:** Link repo public của bạn.
2. [ ] **Thư mục `submission/screenshots/`:** Chứa đủ 4 file ảnh trên.
3. [ ] **File `REFLECTION.md`:** (Nếu yêu cầu) Ghi lại các tham số tốt nhất bạn tìm được ở Bước 1 và các khó khăn đã gặp.

**Chúc bạn hoàn thành tốt bài lab!**
