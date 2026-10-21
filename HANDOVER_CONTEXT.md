# 📋 TÀI LIỆU BÀN GIAO DỰ ÁN (PROJECT HANDOVER)
**Tên Ứng Dụng**: Defect & Dataset Studio AI  
**Thư mục làm việc**: `c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect`  
**Ngày cập nhật**: 15/10/2026  
**Ngôn ngữ & Framework**: Python 3.13, PyQt6, OpenCV (cv2), NumPy, PyInstaller  

---

## 🎯 1. TỔNG QUAN DỰ ÁN
**Defect & Dataset Studio AI** là bộ công cụ chuyên dụng toàn diện dành cho thị giác máy tính công nghiệp (Computer Vision) và huấn luyện mô hình YOLO (v8, v11, Seg, Pose).  
Ứng dụng bao gồm 6 Tab chức năng chính:

1. **Tab 1: Recipe & Zone Setup**
   - Thiết lập ảnh chuẩn (Master Image).
   - Hệ thống căn chỉnh vị trí tự động PRS (Position Reference System) gồm 2 điểm mốc: PRS 1 và PRS 2 (Search ROI + Pattern ROI).
   - Vẽ và quản lý các vùng tạo lỗi: **Defect Zone** (hình chữ nhật hoặc đa giác Polygon).
   - Vẽ các vùng loại trừ: **Ignore Zone** (vùng không cho phép vẽ lỗi).
   - Định nghĩa Keypoints (cho YOLO-Pose).
   - Lưu và nạp Recipe dưới dạng file JSON (`1.json`, `2.json`...).

2. **Tab 2: Batch Generate Defects (Sinh khuyết tật nhân tạo & Xuất YOLO)**
   - Bố cục giao diện: **Chia đôi đều 50/50** (Cột trái: Cài đặt; Cột phải: ImageViewer review to rõ + Terminal Log có nút bật/tắt `📟 Ẩn/Hiện Log`).
   - Cột cài đặt phân thành 2 Sub-tab:
     - `📁 Nguồn & Nhãn`: Input/Output folder, xuất định dạng ảnh, xuất nhãn YOLO (`.txt` BBox hoặc Segmentation), cấu hình **Tự động dò bề mặt Trắng / Đen (Auto Surface Detection)**.
     - `🖌 Cọ vẽ & Lỗi`: Preset, 10 loại cọ lỗi, chế độ hòa trộn (`multiply`, `darken`, `lighten`), các thông số `[Min] ~ [Max]` (Size, Opacity, Intensity, Count, Length, Angle, Main Thick, Sub-Branch, Sub Scale), Thư viện Mẫu thực tế (Stamp Bank), Phạm vi Count (`per_image` vs `per_zone`).
   - Các tính năng cao cấp đã hoàn thiện:
     - **Tự động phát hiện bề mặt Trắng / Đen**: Tự dò vùng sáng (`bright`), vùng tối (`dark`) hoặc cả hai (`both`) bằng Otsu hoặc Manual Threshold, tự động co lùi viền an toàn (`Safety Margin`) để vết lỗi không bao giờ lem ra ngoài mép vật liệu.
     - **Procedural Fiber Brush**: Mô phỏng sợi vải, sợi tóc, bụi cong xoắn tự nhiên vắt ngang bề mặt.
     - **Stamp Bank**: Nạp mẫu khuyết tật thực tế dạng hạt đơn (`scatter`), hỗn hợp (`mixed`) hoặc cụm bụi (`cluster`).

3. **Tab 3: Manual YOLO Labeling (Gán & Chỉnh sửa nhãn thủ công)**
   - Trình gán nhãn thủ công hỗ trợ Bounding Box, Segmentation Polygon, và Pose Keypoints.
   - Hỗ trợ đổi class, xóa nhãn, lưu nhãn nhanh, hiển thị danh sách nhãn dạng bảng.
   - **Tính năng mới**: **`🗑️ Xóa file ảnh & nhãn (Phím tắt Del)`**:
     - Cho phép xóa vĩnh viễn file ảnh và file nhãn (`.txt`) tương ứng của file đang xem khỏi đĩa.
     - Tự động nhận diện thông minh: Nếu đang chọn một annotation cụ thể (BBox/Polygon/Keypoint) trên canvas, phím `Del` sẽ xóa annotation đó; nếu không chọn annotation nào hoặc đang duyệt danh sách file, phím `Del` sẽ kích hoạt xóa file ảnh & nhãn.
     - Hộp thoại xác nhận an toàn (tránh bấm nhầm) hiển thị rõ tên file ảnh & file nhãn sẽ bị xóa, kèm tùy chọn *"Không hỏi lại trong phiên làm việc này"* giúp duyệt lọc dữ liệu nhanh.
     - Tự động chuyển mượt mà sang ảnh kế tiếp sau khi xóa.

4. **Tab 4: Auto PRS & Labeling (Seg / Pose) (Căn chỉnh PRS & Tạo nhãn tự động)**
   - Bố cục giao diện: **Chia đôi đều 50/50** (Cột trái: Cài đặt; Cột phải: ImageViewer hiển thị contour/keypoints + Terminal Log có nút ẩn/hiện).
   - Cột cài đặt phân thành 2 Sub-tab:
     - `📁 Dữ liệu & PRS`: Thư mục input/output, trạng thái định vị PRS 1 & 2, mở rộng ROI margin, tùy chọn lấy trực tiếp đường viền Defect Zone làm nhãn Seg, danh sách file ảnh để click preview.
     - `🪄 Cấu hình Nhãn`: Combobox chọn class nhãn (các nút Thêm/Xóa/Xóa hết cùng hàng), chuyển đổi kiểu nhãn `YOLO-Seg` và `YOLO-Pose`. Dò biên Polarity (Đen→Trắng, Trắng→Đen, Bất kỳ), Hướng tìm (Ngoài→Trong, Trong→Ngoài), khoảng cách tìm kiếm tối đa.
   - Nút Preview nhanh 1 ảnh và Chạy tự động hàng loạt toàn bộ thư mục.

5. **Tab 5: Batch Crop Zones (Cắt trích xuất hàng loạt vùng linh kiện)**
   - Căn chỉnh PRS tự động trên từng ảnh và crop cắt chính xác các vùng linh kiện/Defect Zone đã vẽ ở Tab 1.
   - Xuất các file ảnh đã crop theo thư mục từng zone.

6. **Tab 6: Dataset Manager (Quản lý, Phân tích & Ghép chia tập dữ liệu YOLO)**
   - **Subtab 1 - Dataset Explorer**:
     - Quét (Scan) và khảo sát toàn diện 1 hoặc nhiều dataset nguồn.
     - Bảng thống kê số lượng ảnh, số vật thể, số ảnh background.
     - Bảng phân bố Class (ID, Tên, Số vật thể, %, Số ảnh chứa) kèm màu sắc trực quan.
     - **Tính năng mới**: **`🗑️ Xóa Class đã chọn hàng loạt`** — Chỉ áp dụng đối với thư mục dataset đang chọn trong danh sách nguồn, có **hộp thoại xác nhận bảo mật chi tiết** (hỏi lại trước khi xóa để chống bấm nhầm), tùy chọn tự động đánh lại số ID liên tục (Re-index 0..N-1), cập nhật `data.yaml`, tùy chọn xóa ảnh rỗng và sao lưu `.bak`.
     - Bộ lọc tìm kiếm ảnh theo tên và theo class.
     - Trình xem ảnh YOLO Viewer kèm bảng chi tiết tọa độ từng vật thể.
   - **Subtab 2 - Dataset Builder**:
     - Lựa chọn nhiều nguồn dataset và đặt tỉ lệ lấy riêng từng nguồn (%).
     - Chế độ lấy mẫu: Tỉ lệ %, lấy 100%, hoặc đặt số lượng cụ thể từng class (Target Counts).
     - Tự động phân chia tập Train / Val / Test theo % người dùng đặt.
     - Tự động sinh file `data.yaml` chuẩn tương thích 100% với Ultralytics YOLOv8/v11.

---

## 📂 2. CẤU TRÚC FILE SOURCE CODE
- `main.py`: File khởi chạy ứng dụng (`QApplication(sys.argv)` -> `MainWindow`).
- `main_window.py`: File giao diện chính chứa toàn bộ 6 Tab và các sự kiện UI.
- `brush_engine.py`: Engine sinh khuyết tật thuật toán & mẫu thực tế (Scratch, Spot, Smear, Crack, Burr, Smudge, Stamp, Organic, Fiber, Auto Surface Detection & Safety Margin clipping).
- `custom_widgets.py`: Các widget tùy biến (ImageViewer hỗ trợ zoom/pan/vẽ rect/vẽ polygon/chấm keypoint, ResizableRectItem, ResizablePolygonItem).
- `worker.py`: Các luồng chạy nền (QThread) xử lý đa luồng mượt mà:
  - `DefectWorkerThread`: Tạo defect hàng loạt (Tab 2).
  - `AutoPrsSegWorkerThread`: Dò PRS và auto label seg/pose (Tab 4).
  - `BatchCropWorkerThread`: Cắt trích xuất zone hàng loạt (Tab 5).
  - `DatasetScanWorkerThread`: Quét phân tích dataset (Tab 6).
  - `DatasetBuilderWorkerThread`: Tạo dataset mới và chia train/val/test (Tab 6).
- `utils.py`: Các hàm tiện ích (chuyển đổi base64, parse nhãn YOLO, `cv2_imread`/`cv2_imwrite` xử lý đường dẫn tiếng Việt Unicode).
- `Defect_Dataset_Studio_AI.spec`: File cấu hình PyInstaller để đóng gói ra file `.exe`.
- `assets/`: Thư mục chứa tài nguyên ảnh mẫu khuyết tật thực tế (`stamps.png`).
- `dist/Defect & Dataset Studio AI/`: Thư mục chứa bản build `.exe` đã đóng gói hoàn chỉnh.

---

## 🛠️ 3. CÁCH KIỂM TRA & ĐÓNG GÓI ỨNG DỤNG

### Chạy ứng dụng trực tiếp từ source:
```bash
python main.py
```

### Kiểm tra cú pháp nhanh:
```bash
python -m py_compile main_window.py brush_engine.py worker.py custom_widgets.py
```

### Đóng gói ra file thực thi (.exe) bằng PyInstaller:
```bash
pyinstaller Defect_Dataset_Studio_AI.spec --noconfirm
# Sau khi build xong, luôn đảm bảo thư mục assets được copy sang dist:
powershell -Command "Copy-Item -Path 'assets' -Destination 'dist/Defect & Dataset Studio AI/assets' -Recurse -Force"
```
File thực thi nằm tại: `dist/Defect & Dataset Studio AI/Defect & Dataset Studio AI.exe`.

---

## 💡 4. PROMPT BÀN GIAO SẴN CHO ACCOUNT MỚI
*(Chỉ cần copy & paste đoạn dưới đây vào khung chat của account mới để tiếp tục làm việc ngay lập tức)*

```text
Xin chào! Tôi vừa chuyển sang account mới để tiếp tục phát triển dự án "Defect & Dataset Studio AI" tại thư mục:
c:\Users\ASUS\Desktop\WPy64-31502\APP\Auto Create Defect

Hãy đọc kỹ file HANDOVER_CONTEXT.md trong thư mục trên để nắm toàn bộ cấu trúc 6 tab, các tính năng mới nhất đã làm, và các quy tắc thiết kế giao diện (PyQt6, OpenCV, YOLOv8/v11). Sau khi đọc xong, hãy xác nhận ngắn gọn rằng bạn đã sẵn sàng tiếp nhận dự án.
```
