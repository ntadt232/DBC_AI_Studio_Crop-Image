# 📋 TÀI LIỆU BÀN GIAO DỰ ÁN (PROJECT HANDOVER)
**Tên Ứng Dụng**: Defect & Dataset Studio AI (DBC AI Studio Crop-Image)  
**Thư mục làm việc**: `c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect`  
**Git Repository**: [https://github.com/ntadt232/DBC_AI_Studio_Crop-Image](https://github.com/ntadt232/DBC_AI_Studio_Crop-Image) (Branch: `main`)  
**Ngày cập nhật**: 24/10/2026  
**Ngôn ngữ & Framework**: Python 3.13, PyQt6, OpenCV (cv2), NumPy, PyInstaller  

---

## 🎯 1. TỔNG QUAN DỰ ÁN
**Defect & Dataset Studio AI** là bộ phần mềm hoàn chỉnh chuyên dụng dành cho thị giác máy tính công nghiệp (Computer Vision) và huấn luyện mô hình YOLO (v8, v11, Detect BBox, Seg Polygon, Pose Keypoints).  
Ứng dụng bao gồm 6 Tab chức năng chính:

### 1. Tab 1: Recipe & Zone Setup
- Thiết lập ảnh chuẩn (**Master Image**).
- Hệ thống căn chỉnh vị trí tự động **PRS (Position Reference System)** gồm 2 điểm mốc độc lập: **PRS 1** và **PRS 2** (mỗi PRS gồm Search ROI + Pattern ROI).
- Vẽ và quản lý các vùng tạo lỗi: **Defect Zone** (hỗ trợ cả hình chữ nhật Rect và đa giác tự do Polygon).
- Vẽ các vùng loại trừ: **Ignore Zone** (vùng cấm vẽ lỗi, ví dụ mép IC, chân pin nhạy cảm).
- Định nghĩa danh sách Keypoints (mẫu khung xương cho bài toán YOLO-Pose).
- Lưu và nạp toàn bộ cấu hình Recipe dưới dạng file JSON (`1.json`, `2.json`...).

### 2. Tab 2: Batch Generate Defects (Sinh khuyết tật nhân tạo & Xuất YOLO)
- **Bố cục giao diện**: Chia đều tỉ lệ **50/50** (Cột trái: Cài đặt; Cột phải: ImageViewer review to rõ + Terminal Log có nút bật/tắt `📟 Ẩn/Hiện Log`).
- **Phân nhóm cài đặt thành 2 Sub-tab**:
  - `📁 Nguồn & Nhãn`: Input/Output folder, xuất định dạng ảnh (PNG, BMP, JPG, TIF), xuất nhãn YOLO (`.txt` BBox hoặc Segmentation), cấu hình **Tự động dò bề mặt Trắng / Đen (Auto Surface Detection)**.
  - `🖌 Cọ vẽ & Lỗi`: Preset, 10 loại cọ lỗi, chế độ hòa trộn (`multiply`, `darken`, `lighten`), các thông số `[Min] ~ [Max]` (Size, Opacity, Intensity, Count, Length, Angle, Main Thick, Sub-Branch, Sub Scale), Thư viện Mẫu thực tế (Stamp Bank), Phạm vi Count (`per_image` vs `per_zone`).
- **Các tính năng cao cấp đã hoàn thiện**:
  - **Tự động phát hiện bề mặt Trắng / Đen**: Tự dò vùng sáng (`bright`), vùng tối (`dark`) hoặc cả hai (`both`) bằng Otsu hoặc Manual Threshold, tự động co lùi viền an toàn (`Safety Margin`) để vết lỗi không bao giờ lem ra ngoài mép vật liệu.
  - **Procedural Fiber Brush**: Mô phỏng sợi vải, sợi tóc, bụi cong xoắn tự nhiên vắt ngang bề mặt.
  - **Stamp Bank**: Nạp mẫu khuyết tật thực tế dạng hạt đơn (`scatter`), hỗn hợp (`mixed`) hoặc cụm bụi (`cluster`).

### 3. Tab 3: Manual YOLO Labeling (Gán & Chỉnh sửa nhãn thủ công)
- Trình gán nhãn thủ công hỗ trợ **Bounding Box**, **Segmentation Polygon**, và **Pose Keypoints**.
- Hỗ trợ đổi class, xóa nhãn, lưu nhãn đè file `.txt`, hiển thị danh sách nhãn dạng bảng.
- **Tính năng mới hoàn thiện**: **`🗑️ Xóa file ảnh & nhãn (Phím tắt Del)`**:
  - Cho phép xóa vĩnh viễn cả file ảnh và file nhãn (`.txt`) tương ứng của file đang xem khỏi đĩa.
  - **Xử lý phím Del thông minh**:
    - Nếu đang chọn một annotation cụ thể (BBox/Polygon/Keypoint) trên canvas, phím `Del` sẽ xóa annotation đó.
    - Nếu không chọn annotation nào hoặc đang duyệt danh sách file `list_yolo_images`, phím `Del` sẽ kích hoạt xóa file ảnh & nhãn trên đĩa.
  - Có nút màu đỏ trực quan `🗑️ Xóa file ảnh & nhãn (Del)` ngay dưới danh sách file.
  - Hộp thoại xác nhận an toàn hiển thị chi tiết tên file ảnh & nhãn sẽ xóa, mặc định nút "Hủy bỏ" chống bấm nhầm, kèm tùy chọn *"Không hỏi lại trong phiên làm việc này"* giúp lọc dữ liệu cực nhanh.
  - Tự động chuyển tiếp mượt mà sang ảnh kế tiếp sau khi xóa.

### 4. Tab 4: Auto PRS & Labeling (Seg / Pose) (Căn chỉnh PRS & Tạo nhãn tự động)
- **Bố cục giao diện**: Chia đôi đều **50/50** (Cột trái: Cài đặt; Cột phải: ImageViewer hiển thị contour/keypoints + Terminal Log có nút ẩn/hiện).
- **Phân nhóm cài đặt thành 2 Sub-tab**:
  - `📁 Dữ liệu & PRS`: Thư mục input/output, trạng thái định vị PRS 1 & 2, mở rộng ROI margin, tùy chọn lấy trực tiếp đường viền Defect Zone làm nhãn Seg, danh sách file ảnh để click preview.
  - `🪄 Cấu hình Nhãn`: Combobox chọn class nhãn (các nút Thêm/Xóa/Xóa hết cùng hàng), chuyển đổi kiểu nhãn `YOLO-Seg` và `YOLO-Pose`. Dò biên Polarity (Đen→Trắng, Trắng→Đen, Bất kỳ), Hướng tìm (Ngoài→Trong, Trong→Ngoài), khoảng cách tìm kiếm tối đa.
- Nút Preview nhanh 1 ảnh và Chạy tự động hàng loạt toàn bộ thư mục.

### 5. Tab 5: Batch Crop Zones (Cắt trích xuất hàng loạt vùng linh kiện)
- Căn chỉnh PRS tự động trên từng ảnh và crop cắt chính xác các vùng linh kiện/Defect Zone đã vẽ ở Tab 1.
- Xuất các file ảnh đã crop theo thư mục từng zone phục vụ phân loại bài toán Classification.

### 6. Tab 6: Dataset Manager (Quản lý, Phân tích & Ghép chia tập dữ liệu YOLO)
- **Subtab 1 - Dataset Explorer**:
  - Quét (Scan) và khảo sát toàn diện 1 hoặc nhiều dataset nguồn.
  - Bảng thống kê số lượng ảnh, số vật thể, số ảnh background.
  - Bảng phân bố Class (ID, Tên, Số vật thể, %, Số ảnh chứa) kèm màu sắc trực quan.
  - **Chức năng Xóa Class hàng loạt (`🗑️ Xóa Class đã chọn hàng loạt`)**:
    - Chỉ áp dụng đối với thư mục dataset đang chọn trong danh sách nguồn.
    - Có hộp thoại xác nhận bảo mật chi tiết (chống bấm nhầm).
    - Tự động đánh lại số ID liên tục (Re-index 0..N-1) khi xóa class ở giữa.
    - Đồng bộ cập nhật `data.yaml`, tùy chọn xóa ảnh rỗng (images không còn nhãn) và sao lưu `.bak`.
  - Bộ lọc tìm kiếm ảnh theo tên và theo class, xem chi tiết tọa độ từng vật thể.
- **Subtab 2 - Dataset Builder**:
  - Lựa chọn nhiều nguồn dataset và đặt tỉ lệ lấy riêng từng nguồn (%).
  - Chế độ lấy mẫu: Tỉ lệ %, lấy 100%, hoặc đặt số lượng cụ thể từng class (Target Counts).
  - Tự động phân chia tập Train / Val / Test theo % người dùng đặt.
  - Tự động sinh file `data.yaml` chuẩn tương thích 100% với Ultralytics YOLOv8/v11.

---

## 📂 2. CẤU TRÚC FILE SOURCE CODE

| File / Thư mục | Mục đích & Trách nhiệm |
| :--- | :--- |
| `main.py` | Entry point khởi chạy ứng dụng (`QApplication` -> `MainWindow`). |
| `main_window.py` | Controller chính của ứng dụng chứa toàn bộ logic 6 Tab và tương tác UI. |
| `brush_engine.py` | Thuật toán sinh khuyết tật đồ họa (Scratch, Spot, Smear, Crack, Burr, Smudge, Stamp, Organic, Fiber, Auto Surface Detection & Safety Margin clipping). |
| `custom_widgets.py` | Các widget PyQt6 tùy biến: `ImageViewer` (zoom/pan, vẽ rect/poly/keypoint, xử lý phím Del), `ResizableRectItem`, `ResizablePolygonItem`. |
| `worker.py` | Các tiến trình chạy nền `QThread` xử lý tác vụ nặng không gây đơ giao diện (`DefectWorkerThread`, `AutoPrsSegWorkerThread`, `BatchCropWorkerThread`, `DatasetScanWorkerThread`, `DatasetBuilderWorkerThread`). |
| `utils.py` | Tiện ích hỗ trợ: đọc ghi ảnh hỗ trợ Unicode (`cv2_imread`, `cv2_imwrite`), parse nhãn YOLO, chuyển đổi Base64. |
| `Defect_Dataset_Studio_AI.spec` | Cấu hình PyInstaller dùng để đóng gói ra file `.exe`. |
| `.gitignore` | Loại trừ cache Python (`__pycache__`), thư mục build nặng (`build/`, `dist/`), logs, files tạm. |
| `requirements.txt` | Danh sách thư viện phụ thuộc (`PyQt6`, `opencv-python`, `numpy`). |
| `README.md` | Tài liệu giới thiệu dự án chuẩn trên GitHub. |
| `assets/stamps/` | Các mẫu texture khuyết tật thực tế dùng cho Stamp Bank. |
| `dist/Defect & Dataset Studio AI/` | Thư mục chứa bản build `.exe` đã đóng gói hoàn chỉnh. |

---

## 🛠️ 3. CÁC LỆNH VẬN HÀNH, TEST & ĐÓNG GÓI

### 1. Chạy mã nguồn trực tiếp:
```powershell
Set-Location "c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect"
python main.py
```

### 2. Kiểm tra lỗi cú pháp (Syntax Check):
```powershell
Set-Location "c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect"
python -m py_compile main.py main_window.py brush_engine.py worker.py custom_widgets.py utils.py
```

### 3. Đóng gói ra file `.exe` bằng PyInstaller:
```powershell
Set-Location "c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect"
pyinstaller Defect_Dataset_Studio_AI.spec --noconfirm
Copy-Item -Path "assets" -Destination "dist/Defect & Dataset Studio AI/assets" -Recurse -Force
```
*File thực thi sau build*: `dist/Defect & Dataset Studio AI/Defect & Dataset Studio AI.exe`.

### 4. Đồng bộ Git Repository:
```powershell
Set-Location "c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect"
git status
git add .
git commit -m "commit message"
git push origin main
```

---

## 💡 4. PROMPT BÀN GIAO SẴN CHO ACCOUNT MỚI
*(Chỉ cần copy & paste đoạn dưới đây vào khung chat của account mới để tiếp tục phát triển ngay lập tức)*

```text
Xin chào! Tôi chuyển sang account mới để tiếp tục phát triển dự án "Defect & Dataset Studio AI" tại thư mục:
c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect
GitHub: https://github.com/ntadt232/DBC_AI_Studio_Crop-Image (branch main)

Hãy đọc kỹ file HANDOVER_CONTEXT.md trong thư mục trên để nắm toàn bộ cấu trúc 6 tab, các tính năng mới nhất đã làm (giao diện 50/50 ở Tab 2 & 4, phím Del xóa file ảnh + nhãn ở Tab 3, xóa class hàng loạt ở Tab 6, PRS 1 & 2, procedural brushes).
Sau khi đọc xong, hãy xác nhận ngắn gọn rằng bạn đã sẵn sàng tiếp nhận dự án.
```
