import base64
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage


def img_to_b64(img_arr):
    """Chuyển đổi mảng numpy ảnh sang chuỗi mã hóa Base64"""
    if img_arr is None or getattr(img_arr, 'size', 0) == 0:
        return ""
    if len(img_arr.shape) >= 2 and (img_arr.shape[0] <= 0 or img_arr.shape[1] <= 0):
        return ""
    try:
        success, buffer = cv2.imencode('.png', img_arr)
        if not success or buffer is None:
            return ""
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return ""


def b64_to_img(b64_str):
    """Chuyển đổi chuỗi mã hóa Base64 ngược lại thành mảng numpy ảnh"""
    if not b64_str:
        return None
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


def numpy_to_qpixmap(img):
    """Chuyển đổi ảnh OpenCV (Numpy array) sang QPixmap để hiển thị lên giao diện PyQt6"""
    if img is None:
        return QPixmap()
    img = np.ascontiguousarray(img)
    if len(img.shape) == 2:  # Ảnh xám
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    else:  # Ảnh màu (RGB)
        h, w, ch = img.shape
        qimg = QImage(img.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def cv2_imread(filepath, flags=cv2.IMREAD_COLOR):
    """Đọc ảnh hỗ trợ đường dẫn Unicode trên Windows"""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def cv2_imwrite(filepath, img, params=None):
    """Ghi ảnh hỗ trợ đường dẫn Unicode trên Windows"""
    try:
        import os
        ext = os.path.splitext(filepath)[1]
        if not ext:
            ext = '.png'
        success, buf = cv2.imencode(ext, img, params)
        if success:
            with open(filepath, 'wb') as f:
                f.write(buf)
            return True
        return False
    except Exception:
        return False


def parse_yolo_line(line_str, default_mode='auto', num_kpts=None):
    """
    Phân tích 1 dòng annotation YOLO (hỗ trợ BBox, Pose, Polygon).
    Returns: dict {'type': 'bbox'|'pose'|'poly', 'data': list[float]} hoặc None
    """
    if not line_str:
        return None
    line_str = line_str.strip()
    if not line_str:
        return None
    parts = line_str.split()
    try:
        floats = [float(x) for x in parts]
    except ValueError:
        return None

    if len(parts) == 5:
        return {'type': 'bbox', 'data': floats}

    if len(parts) > 5:
        if num_kpts and len(parts) == 5 + (num_kpts * 3):
            return {'type': 'pose', 'data': floats}
        if default_mode == 'pose' and (len(parts) - 5) % 3 == 0:
            return {'type': 'pose', 'data': floats}
        if (len(parts) - 5) % 3 == 0 and (len(parts) - 5) >= 3:
            vis_flags = [floats[i] for i in range(7, len(floats), 3)]
            if any(v > 1.05 for v in vis_flags) or all(abs(v - round(v)) < 1e-3 and round(v) in (0, 1, 2) for v in vis_flags):
                return {'type': 'pose', 'data': floats}
        return {'type': 'poly', 'data': floats}
    return None
