# FitMotion AI 🏋️

> **Đồ án tốt nghiệp** – Xây dựng hệ thống mô phỏng các bài tập thể chất ứng dụng AI nhận diện tư thế và chia sẻ giáo án tập luyện.

| Thông tin | |
|---|---|
| **Sinh viên** | Phạm Đức Thuận – MSSV: 22050062 – Lớp: 25TH02 |
| **Giảng viên hướng dẫn** | ThS. Nguyễn Hồ Hải |
| **Trường** | Trường Đại Học Bình Dương – Khoa CNTT, Robot và Trí Tuệ Nhân Tạo |

---

## Tính năng chính

- 🎥 **Nhận diện tư thế realtime** qua webcam (YOLOv8-Pose)
- 🦾 **3D Mixamo avatar** hướng dẫn động tác trực quan
- 📊 **Đếm rep tự động** với phản hồi âm thanh tiếng Việt
- 🚨 **Phát hiện ngã & cảnh báo khẩn cấp** (Fall Detection)
- 📅 **Giáo án tập luyện cá nhân hóa** theo mục tiêu
- 👨‍💼 **Admin panel** thêm bài tập & huấn luyện mô hình AI mới
- ⚡ **~20 FPS** trên CPU (imgsz=416, tối ưu latency)

---

## Yêu cầu hệ thống

| Phần mềm | Phiên bản |
|---|---|
| Python | 3.9 – 3.12 |
| MySQL | 8.0 *(tuỳ chọn – nếu không có sẽ dùng SQLite)* |
| Webcam | USB hoặc tích hợp |
| OS | Windows 10/11, Ubuntu 20.04+ |

---

## Cài đặt nhanh

### 1. Clone repo & cài thư viện

```bash
git clone https://github.com/DucThuan12/Doantotnghiep.PhamDucThuan-22050062.git
cd Doantotnghiep.PhamDucThuan-22050062

pip install -r requirements.txt
pip install pymysql   # nếu dùng MySQL
```

> **Lưu ý:** File `.glb` 3D avatar được lưu qua **Git LFS** – cần cài [Git LFS](https://git-lfs.github.com/) rồi chạy `git lfs pull` sau khi clone.

### 2. Cấu hình (tuỳ chọn)

Tạo file `.env` từ mẫu:

```bash
cp .env.example .env
# Chỉnh sửa .env với thông tin MySQL của anh/chị
```

Nếu **không dùng MySQL**, bỏ qua bước này – app tự dùng SQLite.

### 3. Tạo database MySQL (nếu dùng MySQL)

```sql
CREATE DATABASE fitmotion_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Rồi import schema:

```bash
mysql -u root -p fitmotion_ai < scripts/fitmotion_ai_schema.sql
```

### 4. Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt: **http://localhost:5000**

---

## Cấu trúc thư mục

```
├── app.py                  # Flask app chính (5500+ dòng)
├── config.py               # Cấu hình YOLO, camera
├── train.py                # Script huấn luyện pose classifier
├── pose_learning.py        # ML classifier runtime
├── workoutlogic.py         # Logic đếm rep, phát hiện lỗi kỹ thuật
├── emergency.py            # Phát hiện ngã & cảnh báo
├── tts_service.py          # Text-to-Speech tiếng Việt
├── yolov8n-pose.pt         # Model YOLOv8-Pose (6.5 MB)
├── static/
│   ├── audio/              # File âm thanh TTS
│   └── uploads/
│       ├── fbx/            # 3D avatar GLB (Git LFS)
│       └── references/     # Reference motion JSON
├── templates/              # HTML templates (Jinja2)
├── data/
│   └── pose_training/      # Dataset huấn luyện (không được commit)
└── requirements.txt
```

---

## Tài khoản mặc định (sau khi chạy lần đầu)

| Loại | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| User thường | Đăng ký qua UI | — |

---

## Benchmark hiệu năng

| Cấu hình | FPS | Latency E2E |
|---|---|---|
| Baseline (imgsz=640) | 12.14 FPS | 82.64 ms |
| **Tối ưu (imgsz=416)** | **19.88 FPS** | **50.45 ms** |
| Cải thiện | **+63.7%** | **−38.9%** |

*Đo trên CPU Intel, không có GPU.*

---

## Giấy phép

Dự án phục vụ mục đích học thuật. © 2026 Phạm Đức Thuận.