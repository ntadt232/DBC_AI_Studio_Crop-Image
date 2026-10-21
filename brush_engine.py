"""
Brush Engine - Tạo các loại defect tổng hợp cho ảnh sản phẩm công nghiệp.

Hỗ trợ 8 loại brush:
  1. Scratch (vết xước)  - Đường xước thẳng/cong
  2. Spot    (vết đốm)   - Đốm tròn/oval với lõi đậm tán xạ
  3. Smear   (vết lem)   - Vết lem dài, không đều
  4. Crack   (vết nứt)   - Đường nứt phân nhánh
  5. Burr    (vết burr)  - Vết sần sùi, gồ ghề
  6. Smudge  (vết ố mờ)  - Vết xước dạng mảng gồ ghề
  7. Stamp   (mẫu thực)  - Trích xuất từ kho mẫu lỗi thực tế (Natural Defect Stamps)
  8. Organic (hạt hữu cơ)- Mạt bụi/vảy kim loại/rỗ sần procedural với lõi đậm
"""

import os
import sys
import random
import math
import cv2
import numpy as np


# ==========================================
# NGÂN HÀNG MẪU LỖI THỰC TẾ (STAMP BANK)
# ==========================================
class StampBank:
    """
    Quản lý ngân hàng mẫu lỗi thực tế (Natural Defect Stamps) trích xuất từ ảnh tham chiếu.
    Hỗ trợ nạp mặc định từ assets/stamps/ hoặc nạp ảnh tùy chỉnh từ người dùng.
    """
    _default_stamps = []
    _custom_stamps = []
    _custom_source_path = ""
    _initialized = False

    @classmethod
    def get_assets_dir(cls):
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'assets', 'stamps')

    @classmethod
    def extract_stamps_from_image(cls, image_path, min_area=3, max_area=25000):
        stamps = []
        if not os.path.isfile(image_path):
            return stamps
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if im is None:
                im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                return stamps
            
            h, w = im.shape[:2]
            inner = im[3:-3, 3:-3] if (h > 8 and w > 8) else im
            inv = 255 - inner if inner.mean() > 128 else inner
            thresh = (inv > 25).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                area = cv2.contourArea(c)
                if min_area <= area <= max_area:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    x1 = max(0, x - 2)
                    y1 = max(0, y - 2)
                    x2 = min(inner.shape[1], x + w_box + 2)
                    y2 = min(inner.shape[0], y + h_box + 2)
                    crop = inv[y1:y2, x1:x2].copy()
                    # Chỉ giữ đúng 1 hạt: che các hạt lân cận lọt vào khung crop
                    solo = np.zeros_like(crop)
                    cv2.drawContours(solo, [c - np.array([x1, y1])], -1, 255, -1)
                    solo = cv2.dilate(solo, np.ones((3, 3), np.uint8))  # giữ viền mềm
                    crop = cv2.bitwise_and(crop, solo)
                    if crop.size > 0 and crop.max() > 0:
                        stamps.append(crop)
        except Exception as e:
            print(f"Lỗi trích xuất stamps từ {image_path}: {e}")
        return stamps

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        cls._default_stamps = []
        assets_dir = cls.get_assets_dir()
        if os.path.isdir(assets_dir):
            for fname in os.listdir(assets_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    fpath = os.path.join(assets_dir, fname)
                    extracted = cls.extract_stamps_from_image(fpath)
                    cls._default_stamps.extend(extracted)
        cls._initialized = True

    @classmethod
    def load_custom_image(cls, image_path):
        cls._custom_stamps = cls.extract_stamps_from_image(image_path)
        cls._custom_source_path = image_path
        return len(cls._custom_stamps)

    @classmethod
    def reset_to_default(cls):
        cls._custom_stamps = []
        cls._custom_source_path = ""

    @classmethod
    def get_stamps(cls):
        if not cls._initialized:
            cls.initialize()
        if cls._custom_stamps:
            return cls._custom_stamps
        return cls._default_stamps

    @classmethod
    def count(cls):
        return len(cls.get_stamps())


# ==========================================
# CLASS CHÍNH: BRUSH ENGINE
# ==========================================
class BrushEngine:
    """Engine tạo synthetic defect bằng OpenCV drawing + numpy operations"""

    # Hằng số loại brush
    SCRATCH = "scratch"
    SPOT = "spot"
    SMEAR = "smear"
    CRACK = "crack"
    BURR = "burr"
    SMUDGE = "smudge"
    STAMP = "stamp"
    ORGANIC = "organic"
    FIBER = "fiber"

    ALL_TYPES = [SCRATCH, SPOT, SMEAR, CRACK, BURR, SMUDGE, STAMP, ORGANIC, FIBER]

    # Class ID cho YOLO export
    CLASS_MAP = {
        SCRATCH: 0,
        SPOT: 1,
        SMEAR: 2,
        CRACK: 3,
        BURR: 4,
        SMUDGE: 5,
        STAMP: 6,
        ORGANIC: 7,
        FIBER: 8,
    }

    # ==========================================
    # DEFAULT PARAMS
    # ==========================================
    @staticmethod
    def default_params():
        """Trả về tham số mặc định cho brush engine"""
        return {
            'size_min': 3,
            'size_max': 15,
            'opacity_min': 0.3,
            'opacity_max': 0.9,
            'intensity_min': 20,
            'intensity_max': 80,
            'count_min': 1,
            'count_max': 5,
            # --- Tham số mở rộng ---
            'length_min': 30,           # Độ dài tối thiểu (pixels)
            'length_max': 150,          # Độ dài tối đa (pixels)
            'angle_min': 0,             # Hướng vẽ tối thiểu (độ, 0-360)
            'angle_max': 360,           # Hướng vẽ tối đa (độ, 0-360)
            'main_thickness_min': 1,    # Độ dày nhánh chính min
            'main_thickness_max': 5,    # Độ dày nhánh chính max
            'sub_branch_min': 0,        # Số nhánh phụ tối thiểu
            'sub_branch_max': 3,        # Số nhánh phụ tối đa
            'sub_branch_scale': 0.4,    # Tỉ lệ nhánh phụ so với nhánh chính (0.1-1.0)
            'stamp_mode': 'mixed',      # 'mixed', 'scatter', 'cluster'
            'count_scope': 'per_image', # 'per_image' (tổng cả ảnh) hoặc 'per_zone' (mỗi vùng)
        }

    @staticmethod
    def allocate_counts(areas, params):
        """
        Phân bổ số vết lỗi cho từng vùng.
        - per_image: chọn N ∈ [count_min, count_max] cho CẢ ẢNH, rồi chia ngẫu nhiên vào các vùng theo diện tích.
        - per_zone : mỗi vùng tự chọn N ∈ [count_min, count_max] (hành vi cũ).
        Returns: list[int] số vết cho từng vùng (0 = bỏ qua vùng đó).
        """
        n_zones = len(areas)
        if n_zones == 0:
            return []
        cmin = max(1, int(params.get('count_min', 1)))
        cmax = max(cmin, int(params.get('count_max', cmin)))
        if params.get('count_scope', 'per_image') == 'per_zone':
            return [random.randint(cmin, cmax) for _ in range(n_zones)]
        total = random.randint(cmin, cmax)
        weights = [max(1.0, float(a)) for a in areas]
        counts = [0] * n_zones
        for idx in random.choices(range(n_zones), weights=weights, k=total):
            counts[idx] += 1
        return counts

    # ==========================================
    # CÁC HÀM TIỆN ÍCH
    # ==========================================
    @staticmethod
    def _get_valid_start_point(width, height, length, angle):
        """Tính toán điểm bắt đầu (sx, sy) để đoạn thẳng (length, angle) nằm gọn trong khung hình (width, height)"""
        if width <= 0 or height <= 0:
            return 0, 0, 0
            
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Giới hạn chiều dài thực tế tối đa có thể chứa được
        max_lx = int(width / abs(cos_a)) if abs(cos_a) > 0.0001 else float('inf')
        max_ly = int(height / abs(sin_a)) if abs(sin_a) > 0.0001 else float('inf')
        
        actual_length = min(length, max_lx - 1, max_ly - 1)
        if actual_length <= 0:
            return 0, 0, 0
            
        dx = int(actual_length * cos_a)
        dy = int(actual_length * sin_a)
        
        min_x = max(0, -dx)
        max_x = min(width - 1, width - 1 - dx)
        min_y = max(0, -dy)
        max_y = min(height - 1, height - 1 - dy)
        
        # Fallback
        if min_x > max_x: min_x, max_x = 0, width - 1
        if min_y > max_y: min_y, max_y = 0, height - 1
            
        sx = random.randint(min_x, max_x)
        sy = random.randint(min_y, max_y)
        
        return sx, sy, actual_length

    # ==========================================
    # SCRATCH - VẾT XƯỚC
    # ==========================================
    @staticmethod
    def generate_scratch(width, height, params):
        """
        Tạo vết xước: polyline ngẫu nhiên với độ dày biến thiên.
        Sử dụng params: length, angle, main_thickness, sub_branch.
        Returns: (mask, bboxes) - mask uint8, list of [x,y,w,h]
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []
        count = random.randint(params['count_min'], params['count_max'])

        for _ in range(count):
            # Độ dày nhánh chính từ params
            thickness = random.randint(
                max(1, params.get('main_thickness_min', 1)),
                max(2, params.get('main_thickness_max', 5)))

            # Tạo polyline với 3-7 điểm
            num_pts = random.randint(3, 7)

            # Hướng vẽ từ params (chuyển từ độ sang radian)
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            angle = math.radians(random.uniform(angle_min_deg, angle_max_deg))

            # Độ dài từ params
            length = random.randint(
                max(5, params.get('length_min', 30)),
                max(10, params.get('length_max', 150)))

            # Điểm bắt đầu tính toán toán học để đảm bảo chiều dài thực tế
            start_x, start_y, length = BrushEngine._get_valid_start_point(width, height, length, angle)

            points = []
            for i in range(num_pts):
                t = i / max(1, num_pts - 1)
                px = int(start_x + t * length * math.cos(angle) + random.gauss(0, length * 0.03))
                py = int(start_y + t * length * math.sin(angle) + random.gauss(0, length * 0.03))
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))
                points.append([px, py])

            pts = np.array(points, dtype=np.int32)

            # Vẽ từng đoạn để tạo hiệu ứng cành cây (giảm dần độ dày, có độ sần sùi)
            intensity = random.randint(params['intensity_min'], params['intensity_max'])
            current_thickness = float(thickness)
            
            for j in range(len(points) - 1):
                p1 = tuple(points[j])
                p2 = tuple(points[j + 1])
                
                # Biến thiên độ dày: giảm dần về ngọn
                progress = j / (len(points) - 1)
                seg_thick = max(1, int(current_thickness * (1.0 - progress * 0.7)))
                
                # Thêm biến thiên ngẫu nhiên cho bề mặt sần sùi của cành khô
                if random.random() < 0.4:
                    seg_thick += random.randint(0, 1)
                    
                local_int = int(intensity * random.uniform(0.7, 1.0))
                cv2.line(mask, p1, p2, local_int, seg_thick, cv2.LINE_AA)
                current_thickness = max(1.0, current_thickness - random.uniform(0, 0.5))

            # Vẽ nhánh phụ (sub-branches) cành cây khô
            sub_min = params.get('sub_branch_min', 0)
            sub_max = params.get('sub_branch_max', 3)
            sub_scale = params.get('sub_branch_scale', 0.4)
            num_subs = random.randint(sub_min, sub_max)

            for _ in range(num_subs):
                # Chọn điểm phân nhánh (ưu tiên phần thân trước, tránh ngọn)
                branch_idx = random.randint(0, max(0, len(points) - 2))
                bp = points[branch_idx]
                
                # Hướng nhánh phụ (chỉa ra như cành cây)
                sub_angle = angle + random.choice([-1, 1]) * random.uniform(math.pi / 8, math.pi / 3)
                sub_length = int(length * sub_scale * random.uniform(0.4, 0.8))
                
                # Độ dày nhánh phụ nhỏ hơn độ dày nhánh chính tại điểm đó
                base_thick = max(1, int(thickness * (1.0 - (branch_idx / len(points)))))
                sub_thick = max(1, base_thick - 1)
                
                sub_pts_count = random.randint(2, 5)
                sub_points = []
                sp_start = bp
                
                for si in range(sub_pts_count):
                    st = (si + 1) / sub_pts_count
                    sub_angle += random.gauss(0, 0.15)  # Cành cây cong nhẹ không đều
                    seg_len = sub_length / sub_pts_count
                    
                    spx = int(sp_start[0] + seg_len * math.cos(sub_angle))
                    spy = int(sp_start[1] + seg_len * math.sin(sub_angle))
                    spx = max(0, min(width - 1, spx))
                    spy = max(0, min(height - 1, spy))
                    
                    # Vẽ đoạn nhánh phụ (tapering)
                    local_int = int(intensity * random.uniform(0.6, 0.9))
                    st_thick = max(1, int(sub_thick * (1.0 - st * 0.8)))
                    
                    cv2.line(mask, tuple(sp_start), (spx, spy), local_int, st_thick, cv2.LINE_AA)
                    
                    sp_start = [spx, spy]
                    sub_points.append(sp_start)
                    
                points.extend(sub_points)

            # Tính bbox
            all_x = [p[0] for p in points]
            all_y = [p[1] for p in points]
            bx = max(0, min(all_x) - thickness)
            by = max(0, min(all_y) - thickness)
            bw = min(width, max(all_x) + thickness) - bx
            bh = min(height, max(all_y) + thickness) - by
            if bw > 0 and bh > 0:
                bboxes.append([bx, by, bw, bh])

        return mask, bboxes

    # ==========================================
    # SPOT - VẾT ĐỐM
    # ==========================================
    @staticmethod
    def generate_spot(width, height, params):
        """
        Tạo vết đốm: ellipse/circle ngẫu nhiên kèm Gaussian blur.
        Returns: (mask, bboxes)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []
        count = random.randint(params['count_min'], params['count_max'])

        for _ in range(count):
            # Kích thước ellipse
            rx = random.randint(params['size_min'], params['size_max'])
            ry = random.randint(params['size_min'], params['size_max'])

            # Vị trí trung tâm
            cx = random.randint(rx + 2, max(rx + 3, width - rx - 2))
            cy = random.randint(ry + 2, max(ry + 3, height - ry - 2))

            # Góc xoay từ params
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            angle = random.uniform(angle_min_deg, min(angle_max_deg, 360))
            intensity = random.randint(params['intensity_min'], params['intensity_max'])

            # Vẽ ellipse filled
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360,
                        int(intensity), -1, cv2.LINE_AA)

            # Thêm nhiễu texture bên trong
            spot_mask = np.zeros_like(mask)
            cv2.ellipse(spot_mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
            noise = np.random.randint(0, int(intensity * 0.4) + 1,
                                      (height, width), dtype=np.uint8)
            mask = cv2.add(mask, cv2.bitwise_and(noise, spot_mask))

            # Gaussian blur cho mềm mại
            blur_size = max(3, (min(rx, ry) // 2) * 2 + 1)
            region_y1 = max(0, cy - ry * 2)
            region_y2 = min(height, cy + ry * 2)
            region_x1 = max(0, cx - rx * 2)
            region_x2 = min(width, cx + rx * 2)
            if region_y2 > region_y1 and region_x2 > region_x1:
                roi = mask[region_y1:region_y2, region_x1:region_x2]
                mask[region_y1:region_y2, region_x1:region_x2] = cv2.GaussianBlur(
                    roi, (blur_size, blur_size), 0)

            # Bbox
            bx = max(0, cx - rx - 2)
            by = max(0, cy - ry - 2)
            bw = min(width - bx, 2 * rx + 4)
            bh = min(height - by, 2 * ry + 4)
            if bw > 0 and bh > 0:
                bboxes.append([bx, by, bw, bh])

        return mask, bboxes

    # ==========================================
    # SMEAR - VẾT LEM
    # ==========================================
    @staticmethod
    def generate_smear(width, height, params):
        """
        Tạo vết lem: Bézier curve dày, bán trong suốt, cạnh không đều.
        Returns: (mask, bboxes)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []
        count = random.randint(params['count_min'], params['count_max'])

        for _ in range(count):
            # Độ dày nhánh chính từ params
            thickness = random.randint(
                max(1, params.get('main_thickness_min', 1)),
                max(2, params.get('main_thickness_max', 5)) * 2)
            intensity = random.randint(params['intensity_min'], params['intensity_max'])

            # Hướng vẽ từ params
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            main_angle = math.radians(random.uniform(angle_min_deg, angle_max_deg))

            # Độ dài từ params
            smear_length = random.randint(
                max(10, params.get('length_min', 30)),
                max(20, params.get('length_max', 150)))

            # Tính toán điểm bắt đầu
            sx, sy, smear_length = BrushEngine._get_valid_start_point(width, height, smear_length, main_angle)
            p0 = [sx, sy]
            # Điểm cuối theo hướng + độ dài
            end_x = p0[0] + int(smear_length * math.cos(main_angle))
            end_y = p0[1] + int(smear_length * math.sin(main_angle))
            # Control point lệch để tạo đường cong
            ctrl_offset = smear_length * 0.3
            p1 = [int((p0[0] + end_x) / 2 + random.gauss(0, ctrl_offset)),
                   int((p0[1] + end_y) / 2 + random.gauss(0, ctrl_offset))]
            p2 = [max(0, min(width - 1, end_x)),
                   max(0, min(height - 1, end_y))]

            # Sample Bézier curve
            curve_points = []
            num_samples = 30
            for i in range(num_samples + 1):
                t = i / num_samples
                x = int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0])
                y = int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1])
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                curve_points.append([x, y])

            # Vẽ đường lem với thickness biến thiên
            for i in range(len(curve_points) - 1):
                t_var = thickness + random.randint(-thickness // 3, thickness // 3)
                t_var = max(2, t_var)
                local_intensity = int(intensity * random.uniform(0.5, 1.0))
                cv2.line(mask, tuple(curve_points[i]), tuple(curve_points[i + 1]),
                         local_intensity, t_var, cv2.LINE_AA)

            # Gaussian blur cho vết lem mềm
            blur_k = max(3, (thickness // 2) * 2 + 1)
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        # Tính bbox
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 10:
                bboxes.append(cv2.boundingRect(c))

        return mask, bboxes

    # ==========================================
    # SMUDGE - VẾT Ố/CHÀ XÁT (ĐÁM MÂY DÀI)
    # ==========================================
    @staticmethod
    def generate_smudge(width, height, params):
        """
        Tạo vết xước dạng mảng (scrape/chip): Cạnh cực kỳ sắc nét, gồ ghề (blocky/jagged), và mờ dần (gradient fade) về một hướng.
        Returns: (mask, bboxes)
        """
        mask = np.zeros((height, width), dtype=np.float32)
        bboxes = []
        count = random.randint(params['count_min'], params['count_max'])

        for _ in range(count):
            intensity = random.randint(params['intensity_min'], params['intensity_max'])
            length = random.randint(
                max(50, params.get('length_min', 80)),
                max(100, params.get('length_max', 250)))
                
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            main_angle = math.radians(random.uniform(angle_min_deg, angle_max_deg))

            sx, sy, length = BrushEngine._get_valid_start_point(width, height, length, main_angle)
            p0 = [sx, sy]
            end_x = p0[0] + int(length * math.cos(main_angle))
            end_y = p0[1] + int(length * math.sin(main_angle))

            # 1. Tạo đường zigzag (cạnh sắc, dạng bậc thang/blocky)
            num_points = max(10, length // 2)  # Mật độ điểm cao để răng cưa chi tiết
            pts = []
            
            # Vector pháp tuyến (hướng mờ)
            fade_angle = main_angle + math.pi / 2
            if random.random() > 0.5:
                fade_angle += math.pi
                
            nx = math.cos(fade_angle)
            ny = math.sin(fade_angle)
            
            # Hàm random walk có quán tính để tạo độ gồ ghề dạng bậc thang (blocky)
            current_offset = 0.0
            jaggedness = random.uniform(3.0, 10.0)
            
            for i in range(num_points + 1):
                t = i / num_points
                base_x = p0[0] * (1 - t) + end_x * t
                base_y = p0[1] * (1 - t) + end_y * t
                
                # Thỉnh thoảng mới giật bậc (tạo độ sắc/blocky giống ảnh)
                if random.random() > 0.75:  
                    current_offset += random.uniform(-jaggedness, jaggedness)
                    # Giới hạn biên độ răng cưa
                    current_offset = max(-jaggedness*2.5, min(jaggedness*2.5, current_offset))
                
                px = int(base_x - nx * current_offset)  # Trừ nx để offset ngược hướng fade
                py = int(base_y - ny * current_offset)
                pts.append([px, py])

            pts_array = np.array(pts, np.int32)
            
            # 2. Tạo gradient sweep (mờ dần về hướng fade_angle)
            temp_mask = np.zeros((height, width), dtype=np.float32)
            
            # Chiều dài vệt mờ phụ thuộc vào thickness min/max của brush
            t_min = params.get('main_thickness_min', 15)
            t_max = params.get('main_thickness_max', 40)
            fade_distance = random.randint(max(10, t_min), max(20, t_max)) 
            
            for d in range(fade_distance):
                # Tỷ lệ mờ dần (từ 1.0 xuống 0.0)
                alpha = 1.0 - (d / fade_distance)
                # Curve cường độ hàm mũ để gradient tự nhiên (nhạt nhanh ở đuôi)
                alpha = alpha ** 1.5 
                
                current_intensity = intensity * alpha
                
                # Dịch chuyển các điểm theo hướng fade
                shifted_pts = pts_array.copy().astype(np.float32)
                shifted_pts[:, 0] += nx * d
                shifted_pts[:, 1] += ny * d
                
                shifted_pts = shifted_pts.astype(np.int32).reshape((-1, 1, 2))
                
                # Vẽ line sweep lấp đầy khu vực gradient
                cv2.polylines(temp_mask, [shifted_pts], False, current_intensity, 2, cv2.LINE_AA)
                
            # Thêm nhiễu hạt (noise) sần sùi cho bề mặt vết xước
            noise = np.random.normal(0, intensity * 0.15, (height, width))
            temp_mask = temp_mask + noise * (temp_mask > 0)
            
            # Làm mờ nhẹ phần gradient để trộn các line sweep lại với nhau
            temp_mask = cv2.GaussianBlur(temp_mask, (3, 3), 0)
            
            # 3. Vẽ lại cạnh gốc cực kỳ sắc nét trên cùng (leading edge)
            cv2.polylines(temp_mask, [pts_array.reshape((-1, 1, 2))], False, intensity * 1.2, 1, cv2.LINE_AA)

            mask = np.maximum(mask, temp_mask)

        mask = np.clip(mask, 0, 255).astype(np.uint8)

        # Tính bbox
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 20:
                bboxes.append(cv2.boundingRect(c))

        return mask, bboxes

    # ==========================================
    # CRACK - VẾT NỨT
    # ==========================================
    @staticmethod
    def generate_crack(width, height, params):
        """
        Tạo vết nứt: đường phân nhánh fractal-like.
        Returns: (mask, bboxes)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        all_points = []

        count = random.randint(params['count_min'], params['count_max'])
        intensity = random.randint(params['intensity_min'], params['intensity_max'])

        # Tham số nhánh phụ từ params
        sub_branch_min = params.get('sub_branch_min', 0)
        sub_branch_max = params.get('sub_branch_max', 3)
        sub_branch_scale = params.get('sub_branch_scale', 0.4)

        for _ in range(count):
            # Độ dài từ params
            length_req = random.randint(
                max(10, params.get('length_min', 30)),
                max(20, params.get('length_max', 150)))

            # Hướng vẽ từ params
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            main_angle = math.radians(random.uniform(angle_min_deg, angle_max_deg))

            # Tính toán điểm bắt đầu
            sx, sy, main_length = BrushEngine._get_valid_start_point(width, height, length_req, main_angle)
            crack_points = [(sx, sy)]

            # Độ dày nhánh chính từ params
            max_thickness = random.randint(
                max(1, params.get('main_thickness_min', 1)),
                max(2, params.get('main_thickness_max', 5)))

            # Số nhánh phụ cho mỗi segment
            target_sub_branches = random.randint(sub_branch_min, sub_branch_max)
            # Xác suất phân nhánh dựa trên target
            branch_probability = min(0.8, target_sub_branches / 10.0 + 0.1) if target_sub_branches > 0 else 0.0

            def _draw_branch(start, angle, length, thickness, depth, points_list, is_sub=False):
                """Đệ quy vẽ nhánh crack"""
                if depth <= 0 or length < 3 or thickness < 1:
                    return

                # Di chuyển theo hướng với nhiễu
                num_segments = random.randint(3, 8)
                current = list(start)
                seg_len = length / num_segments

                for s in range(num_segments):
                    # Nhiễu góc
                    angle += random.gauss(0, 0.4)
                    nx = int(current[0] + seg_len * math.cos(angle))
                    ny = int(current[1] + seg_len * math.sin(angle))
                    nx = max(0, min(width - 1, nx))
                    ny = max(0, min(height - 1, ny))

                    # Thickness giảm dần
                    seg_thick = max(1, int(thickness * (1 - s / num_segments * 0.6)))
                    local_int = int(intensity * random.uniform(0.6, 1.0))

                    cv2.line(mask, (int(current[0]), int(current[1])), (nx, ny),
                             local_int, seg_thick, cv2.LINE_AA)

                    points_list.append((nx, ny))
                    current = [nx, ny]

                    # Phân nhánh phụ — xác suất dựa trên params
                    if not is_sub and random.random() < branch_probability and depth > 1:
                        branch_angle = angle + random.choice([-1, 1]) * random.uniform(0.4, 1.2)
                        branch_length = length * sub_branch_scale * random.uniform(0.5, 1.0)
                        branch_thick = max(1, thickness - 1)
                        _draw_branch(tuple(current), branch_angle, branch_length,
                                     branch_thick, depth - 1, points_list, is_sub=True)

            _draw_branch((sx, sy), main_angle, main_length, max_thickness, 4, crack_points)
            all_points.extend(crack_points)

        # Tính bbox tổng
        bboxes = []
        if all_points:
            all_x = [p[0] for p in all_points]
            all_y = [p[1] for p in all_points]
            pad = max_thickness + 2 if count > 0 else 2
            bx = max(0, min(all_x) - pad)
            by = max(0, min(all_y) - pad)
            bw = min(width, max(all_x) + pad) - bx
            bh = min(height, max(all_y) + pad) - by
            if bw > 0 and bh > 0:
                bboxes.append([bx, by, bw, bh])

        return mask, bboxes

    # ==========================================
    # BURR - VẾT BURR (sần sùi)
    # ==========================================
    @staticmethod
    def generate_burr(width, height, params):
        """
        Tạo vết burr: khối sần sùi, cạnh gồ ghề với nhiễu texture.
        Returns: (mask, bboxes)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []
        count = random.randint(params['count_min'], params['count_max'])

        for _ in range(count):
            # Kích thước blob
            blob_w = random.randint(params['size_min'] * 2, params['size_max'] * 3)
            blob_h = random.randint(params['size_min'] * 2, params['size_max'] * 3)

            # Vị trí
            cx = random.randint(blob_w, max(blob_w + 1, width - blob_w))
            cy = random.randint(blob_h, max(blob_h + 1, height - blob_h))

            intensity = random.randint(params['intensity_min'], params['intensity_max'])

            # Hướng xoay từ params
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            base_angle_offset = math.radians(random.uniform(angle_min_deg, angle_max_deg))

            # Tạo contour bất quy tắc bằng polar coordinates
            num_vertices = random.randint(8, 20)
            contour_pts = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices + base_angle_offset
                # Bán kính biến thiên ngẫu nhiên
                r_x = blob_w / 2 * random.uniform(0.5, 1.3)
                r_y = blob_h / 2 * random.uniform(0.5, 1.3)
                px = int(cx + r_x * math.cos(angle))
                py = int(cy + r_y * math.sin(angle))
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))
                contour_pts.append([px, py])

            contour = np.array(contour_pts, dtype=np.int32)
            cv2.fillPoly(mask, [contour], int(intensity))

            # Thêm noise texture bên trong
            burr_mask_region = np.zeros_like(mask)
            cv2.fillPoly(burr_mask_region, [contour], 255)
            noise = np.random.randint(0, int(intensity * 0.5) + 1,
                                      (height, width), dtype=np.uint8)
            mask = cv2.add(mask, cv2.bitwise_and(noise, burr_mask_region))

            # Thêm vết gồ ghề ở viền
            edge_thickness = random.randint(1, max(2, params['size_min']))
            for i in range(len(contour_pts)):
                if random.random() < 0.5:
                    j = (i + 1) % len(contour_pts)
                    cv2.line(mask, tuple(contour_pts[i]), tuple(contour_pts[j]),
                             int(intensity * random.uniform(0.8, 1.2)),
                             edge_thickness + random.randint(0, 2), cv2.LINE_AA)

            # Light blur cho tự nhiên
            blur_k = max(3, (min(blob_w, blob_h) // 4) * 2 + 1)
            region_y1 = max(0, cy - blob_h)
            region_y2 = min(height, cy + blob_h)
            region_x1 = max(0, cx - blob_w)
            region_x2 = min(width, cx + blob_w)
            if region_y2 > region_y1 and region_x2 > region_x1:
                roi = mask[region_y1:region_y2, region_x1:region_x2]
                mask[region_y1:region_y2, region_x1:region_x2] = cv2.GaussianBlur(
                    roi, (blur_k, blur_k), 0)

            # Bbox
            all_x = [p[0] for p in contour_pts]
            all_y = [p[1] for p in contour_pts]
            bx = max(0, min(all_x) - 2)
            by = max(0, min(all_y) - 2)
            bw = min(width - bx, max(all_x) - bx + 4)
            bh = min(height - by, max(all_y) - by + 4)
            if bw > 0 and bh > 0:
                bboxes.append([bx, by, bw, bh])

        return mask, bboxes

    # ==========================================
    # STAMP - MẪU LỖI THỰC TẾ (NATURAL DEFECT STAMP)
    # ==========================================
    @staticmethod
    def _transform_stamp(stamp, nw, nh, angle_min=0, angle_max=360):
        """Co giãn, xoay, lật gương và biến dạng affine hữu cơ nhẹ"""
        nw = max(2, int(nw))
        nh = max(2, int(nh))
        resized = cv2.resize(stamp, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # Flip ngẫu nhiên
        if random.random() < 0.5:
            resized = cv2.flip(resized, 1)
        if random.random() < 0.5:
            resized = cv2.flip(resized, 0)

        # Xoay ngẫu nhiên
        angle = random.uniform(angle_min, angle_max)
        M = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        bound_w = max(2, int((nh * sin) + (nw * cos)))
        bound_h = max(2, int((nh * cos) + (nw * sin)))
        M[0, 2] += (bound_w / 2.0) - nw / 2.0
        M[1, 2] += (bound_h / 2.0) - nh / 2.0

        # Biến dạng affine nhẹ (shear) để mỗi lần xuất hiện đều có hình thái mới lạ
        shear_x = random.uniform(-0.12, 0.12)
        shear_y = random.uniform(-0.12, 0.12)
        M[0, 1] += shear_x
        M[1, 0] += shear_y

        transformed = cv2.warpAffine(
            resized, M, (bound_w, bound_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return transformed

    @classmethod
    def generate_stamp(cls, width, height, params):
        """
        Tạo defect từ ngân hàng mẫu lỗi thực tế (Stamp Bank).
        Trích xuất từ các ảnh mẫu thực tế với biến dạng ngẫu nhiên (scale, rotate, flip, shear).
        """
        stamps = StampBank.get_stamps()
        if not stamps:
            return cls.generate_organic(width, height, params)

        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []

        count_min = max(1, params.get('count_min', 1))
        count_max = max(count_min, params.get('count_max', 5))
        count = random.randint(count_min, count_max)

        size_min = max(2, params.get('size_min', 5))
        size_max = max(size_min, params.get('size_max', 25))

        intensity_min = params.get('intensity_min', 120)
        intensity_max = params.get('intensity_max', 255)

        angle_min = params.get('angle_min', 0)
        angle_max = params.get('angle_max', 360)

        stamp_mode = params.get('stamp_mode', 'mixed')
        is_cluster = (stamp_mode == 'cluster') or (stamp_mode == 'mixed' and random.random() < 0.4)

        if is_cluster:
            cx = random.randint(width // 4, max(width // 4 + 1, width * 3 // 4))
            cy = random.randint(height // 4, max(height // 4 + 1, height * 3 // 4))
            spread = max(8.0, min(width, height) * 0.22)

            num_particles = count * random.randint(2, 5)
            cluster_items = []

            for _ in range(num_particles):
                stamp = random.choice(stamps)
                sh, sw = stamp.shape[:2]
                if sh <= 0 or sw <= 0:
                    continue

                is_main = (random.random() < 0.25)
                cur_size = random.randint(size_min, size_max) if is_main else max(2, int(size_min * random.uniform(0.3, 0.85)))
                scale = cur_size / max(sh, sw)
                nw = max(2, int(sw * scale))
                nh = max(2, int(sh * scale))

                transformed = cls._transform_stamp(stamp, nw, nh, angle_min, angle_max)
                if transformed is None or transformed.size == 0 or transformed.max() == 0:
                    continue

                target_int = random.randint(intensity_min, intensity_max)
                transformed = (transformed.astype(np.float32) / transformed.max() * target_int).astype(np.uint8)

                th, tw = transformed.shape
                if tw >= width or th >= height:
                    continue

                px = int(cx + random.gauss(0, spread) - tw // 2)
                py = int(cy + random.gauss(0, spread) - th // 2)
                px = max(0, min(width - tw, px))
                py = max(0, min(height - th, py))

                roi_mask = mask[py:py+th, px:px+tw]
                mask[py:py+th, px:px+tw] = np.maximum(roi_mask, transformed)
                cluster_items.append((px, py, tw, th))

            if cluster_items:
                all_x = [p[0] for p in cluster_items]
                all_y = [p[1] for p in cluster_items]
                all_x2 = [p[0] + p[2] for p in cluster_items]
                all_y2 = [p[1] + p[3] for p in cluster_items]
                bx = max(0, min(all_x))
                by = max(0, min(all_y))
                bw = min(width - bx, max(all_x2) - bx)
                bh = min(height - by, max(all_y2) - by)
                if bw > 0 and bh > 0:
                    bboxes.append([bx, by, bw, bh])
        else:
            for _ in range(count):
                stamp = random.choice(stamps)
                sh, sw = stamp.shape[:2]
                if sh <= 0 or sw <= 0:
                    continue

                cur_size = random.randint(size_min, size_max)
                scale = cur_size / max(sh, sw)
                nw = max(2, int(sw * scale))
                nh = max(2, int(sh * scale))

                transformed = cls._transform_stamp(stamp, nw, nh, angle_min, angle_max)
                if transformed is None or transformed.size == 0 or transformed.max() == 0:
                    continue

                target_int = random.randint(intensity_min, intensity_max)
                transformed = (transformed.astype(np.float32) / transformed.max() * target_int).astype(np.uint8)

                th, tw = transformed.shape
                if tw >= width or th >= height:
                    continue

                px = random.randint(0, max(0, width - tw))
                py = random.randint(0, max(0, height - th))

                roi_mask = mask[py:py+th, px:px+tw]
                mask[py:py+th, px:px+tw] = np.maximum(roi_mask, transformed)

                cnts, _ = cv2.findContours((transformed > 15).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    sbx, sby, sbw, sbh = cv2.boundingRect(np.vstack(cnts))
                    bboxes.append([px + sbx, py + sby, sbw, sbh])

        if not bboxes and np.count_nonzero(mask) > 0:
            cnts, _ = cv2.findContours((mask > 15).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= 3:
                    bboxes.append(list(cv2.boundingRect(c)))

        return mask, bboxes

    # ==========================================
    # ORGANIC - HẠT MẠT BỤI HỮU CƠ (PROCEDURAL)
    # ==========================================
    @classmethod
    def generate_organic(cls, width, height, params):
        """
        Tạo hạt mạt bụi/vảy kim loại/rỗ sần hữu cơ (Procedural Organic Particles).
        Không cần ảnh ngoài: dùng Polar jitter + Morphological fractal edges + Distance Transform Core Density.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []

        count_min = max(1, params.get('count_min', 1))
        count_max = max(count_min, params.get('count_max', 5))
        count = random.randint(count_min, count_max)

        size_min = max(3, params.get('size_min', 5))
        size_max = max(size_min, params.get('size_max', 25))
        intensity_min = params.get('intensity_min', 120)
        intensity_max = params.get('intensity_max', 255)

        for _ in range(count):
            size = random.randint(size_min, size_max)
            intensity = random.randint(intensity_min, intensity_max)

            ph = size * 2 + 8
            pw = size * 2 + 8
            if pw >= width or ph >= height:
                ph = min(height - 1, ph)
                pw = min(width - 1, pw)
                if pw < 4 or ph < 4:
                    continue

            p_mask = np.zeros((ph, pw), dtype=np.float32)
            pcx, pcy = pw // 2, ph // 2

            num_pts = random.randint(10, 24)
            angles = np.linspace(0, 2 * math.pi, num_pts, endpoint=False)
            radii = (size * 0.5) * (0.4 + 0.6 * np.random.rand(num_pts))

            # 25% cơ hội có đuôi kéo dài dạng comet tail
            if random.random() < 0.25:
                tail_idx = random.randint(0, num_pts - 1)
                radii[tail_idx] *= random.uniform(1.6, 2.3)
                if tail_idx + 1 < num_pts:
                    radii[tail_idx + 1] *= 1.4

            pts = np.array([
                [pcx + r * math.cos(a), pcy + r * math.sin(a)]
                for a, r in zip(angles, radii)
            ], dtype=np.int32)
            cv2.fillPoly(p_mask, [pts], 1.0)

            # Nhiễu hình thái học (Morphological close)
            k_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            m_noise = (np.random.rand(ph, pw) > 0.35).astype(np.float32)
            p_mask = cv2.morphologyEx(p_mask * (0.6 + 0.4 * m_noise), cv2.MORPH_CLOSE, kernel)

            # Distance transform cho lõi đậm - viền tán xạ mềm mại
            bin_mask = (p_mask > 0.2).astype(np.uint8)
            if np.count_nonzero(bin_mask) > 0:
                dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
                max_d = dist.max()
                if max_d > 0:
                    dist_norm = dist / max_d
                    optical_density = 0.4 * dist_norm + 0.6 * (dist_norm ** 0.5)
                    p_mask = p_mask * optical_density * intensity
                else:
                    p_mask = p_mask * intensity
            else:
                p_mask = p_mask * intensity

            p_uint8 = np.clip(p_mask, 0, 255).astype(np.uint8)

            pos_x = random.randint(0, max(0, width - pw))
            pos_y = random.randint(0, max(0, height - ph))

            roi = mask[pos_y:pos_y+ph, pos_x:pos_x+pw]
            mask[pos_y:pos_y+ph, pos_x:pos_x+pw] = np.maximum(roi, p_uint8)

            cnts, _ = cv2.findContours((p_uint8 > 15).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                sbx, sby, sbw, sbh = cv2.boundingRect(np.vstack(cnts))
                bboxes.append([pos_x + sbx, pos_y + sby, sbw, sbh])

        if not bboxes and np.count_nonzero(mask) > 0:
            cnts, _ = cv2.findContours((mask > 15).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= 3:
                    bboxes.append(list(cv2.boundingRect(c)))

        return mask, bboxes

    # ==========================================
    # FIBER - SỢI BỤI MẢNH CONG, XOẮN VẮT NGANG
    # ==========================================
    @classmethod
    def generate_fiber(cls, width, height, params):
        """
        Tạo sợi bụi mảnh cong, xoắn vắt ngang qua vật liệu (Fiber / Lint / Dust Thread):
        - Sợi cong uốn lượn hình sin/xoắn tự nhiên với độ dài và hướng ngẫu nhiên
        - Độ dày mảnh vuốt nhọn (tapering) ở 2 đầu
        - Sử dụng params: length, angle, main_thickness, count
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = []

        count_min = max(1, params.get('count_min', 1))
        count_max = max(count_min, params.get('count_max', 3))
        count = random.randint(count_min, count_max)

        for _ in range(count):
            # Chiều dài sợi
            len_min = max(15, params.get('length_min', 30))
            len_max = max(len_min, params.get('length_max', 150))
            fiber_len = random.randint(len_min, len_max)

            # Góc định hướng chung
            angle_min_deg = params.get('angle_min', 0)
            angle_max_deg = params.get('angle_max', 360)
            base_angle = math.radians(random.uniform(angle_min_deg, angle_max_deg))

            # Độ dày tối đa ở thân giữa (mặc định 1 - 3 px)
            thick_max = max(1, random.randint(
                max(1, params.get('main_thickness_min', 1)),
                max(1, params.get('main_thickness_max', 3))
            ))

            # Số điểm trên sợi tỷ lệ với độ dài (mỗi điểm cách ~2-3 px)
            num_steps = max(20, int(fiber_len / 2.0))

            # Chọn kiểu sợi: wavy (sóng sin), loop (vòng xoắn/curl), spline (quán tính tự do)
            fiber_style = random.choice(['wavy', 'loop', 'spline'])

            wave_amp = random.uniform(5.0, max(8.0, fiber_len * 0.15))
            wave_freq = random.uniform(1.0, 3.5)
            wave_phase = random.uniform(0, 2 * math.pi)

            start_x = random.randint(0, max(0, width - 1))
            start_y = random.randint(0, max(0, height - 1))

            pts = []
            curr_x = float(start_x)
            curr_y = float(start_y)
            curr_angle = base_angle

            angle_velocity = random.uniform(-0.15, 0.15)
            step_size = fiber_len / float(num_steps)

            for i in range(num_steps):
                t = i / float(num_steps - 1)  # 0.0 -> 1.0

                if fiber_style == 'wavy':
                    forward = t * fiber_len
                    lateral = math.sin(2 * math.pi * wave_freq * t + wave_phase) * wave_amp
                    px = start_x + forward * math.cos(base_angle) - lateral * math.sin(base_angle)
                    py = start_y + forward * math.sin(base_angle) + lateral * math.cos(base_angle)
                elif fiber_style == 'loop':
                    forward = t * fiber_len
                    loop_factor = math.exp(-((t - 0.5) ** 2) / 0.04)  # Đạt cực đại ở giữa sợi
                    curl_angle = wave_phase + t * 4 * math.pi * loop_factor
                    curl_r = wave_amp * loop_factor
                    px = start_x + forward * math.cos(base_angle) + curl_r * math.cos(curl_angle)
                    py = start_y + forward * math.sin(base_angle) + curl_r * math.sin(curl_angle)
                else:  # 'spline'
                    angle_velocity += random.gauss(0, 0.05)
                    angle_velocity = max(-0.25, min(0.25, angle_velocity))
                    curr_angle += angle_velocity
                    curr_x += step_size * math.cos(curr_angle)
                    curr_y += step_size * math.sin(curr_angle)
                    px = curr_x
                    py = curr_y

                pts.append((px, py))

            # Căn chỉnh để các điểm nằm trọn vẹn trong vùng ảnh
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pw = max_x - min_x
            ph = max_y - min_y

            if width > pw + 4:
                shift_x = random.randint(2, int(width - pw - 2)) - min_x
            else:
                shift_x = -min_x
            if height > ph + 4:
                shift_y = random.randint(2, int(height - ph - 2)) - min_y
            else:
                shift_y = -min_y

            adjusted_pts = [(p[0] + shift_x, p[1] + shift_y) for p in pts]

            intensity = random.randint(
                params.get('intensity_min', 150),
                params.get('intensity_max', 255)
            )

            # Vẽ sợi với tapering (vuốt nhọn 2 đầu bằng hàm sin(t * pi))
            fiber_mask = np.zeros((height, width), dtype=np.uint8)
            for i in range(len(adjusted_pts) - 1):
                t = i / float(len(adjusted_pts) - 1)
                taper = math.sin(t * math.pi)
                seg_thick = max(1, int(round(thick_max * (0.3 + 0.7 * taper))))

                p1 = (int(round(adjusted_pts[i][0])), int(round(adjusted_pts[i][1])))
                p2 = (int(round(adjusted_pts[i+1][0])), int(round(adjusted_pts[i+1][1])))

                cv2.line(fiber_mask, p1, p2, 255, seg_thick, cv2.LINE_AA)

            if thick_max >= 2:
                fiber_mask = cv2.GaussianBlur(fiber_mask, (3, 3), 0.5)

            if intensity < 255:
                fiber_mask = (fiber_mask.astype(np.float32) * (intensity / 255.0)).astype(np.uint8)

            mask = np.maximum(mask, fiber_mask)

            cnts, _ = cv2.findContours((fiber_mask > 15).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                sbx, sby, sbw, sbh = cv2.boundingRect(np.vstack(cnts))
                bboxes.append([sbx, sby, sbw, sbh])

        if not bboxes and np.count_nonzero(mask) > 0:
            cnts, _ = cv2.findContours((mask > 15).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= 3:
                    bboxes.append(list(cv2.boundingRect(c)))

        return mask, bboxes

    # ==========================================
    # MAIN API
    # ==========================================
    @classmethod
    def generate_defect(cls, width, height, brush_type, params=None):
        """
        Tạo defect mask ngẫu nhiên.

        Args:
            width, height: Kích thước vùng defect zone
            brush_type: Loại brush (scratch/spot/smear/crack/burr/smudge/stamp/organic/fiber/random)
            params: Dict tham số (dùng default nếu None)

        Returns:
            (mask, bboxes, actual_brush_type)
            - mask: np.uint8 shape (height, width), giá trị 0-255
            - bboxes: list of [x, y, w, h] local coords
            - actual_brush_type: str loại brush đã dùng
        """
        if params is None:
            params = cls.default_params()

        # Chọn ngẫu nhiên nếu là "random"
        actual_type = brush_type
        if brush_type == "random":
            actual_type = random.choice(cls.ALL_TYPES)

        generators = {
            cls.SCRATCH: cls.generate_scratch,
            cls.SPOT: cls.generate_spot,
            cls.SMEAR: cls.generate_smear,
            cls.CRACK: cls.generate_crack,
            cls.BURR: cls.generate_burr,
            cls.SMUDGE: cls.generate_smudge,
            cls.STAMP: cls.generate_stamp,
            cls.ORGANIC: cls.generate_organic,
            cls.FIBER: cls.generate_fiber,
        }

        gen_func = generators.get(actual_type, cls.generate_scratch)
        mask, bboxes = gen_func(width, height, params)

        return mask, bboxes, actual_type

    @staticmethod
    def apply_defect(image, zone_rect, mask, opacity=0.7, mode="multiply"):
        """
        Áp dụng defect mask lên ảnh tại vị trí zone_rect.

        Args:
            image: np.uint8 ảnh gốc (grayscale hoặc BGR)
            zone_rect: [x, y, w, h] vị trí zone trên ảnh
            mask: np.uint8 defect mask (cùng kích thước với zone)
            opacity: float 0.0-1.0 độ đậm
            mode: "multiply" (tự nhiên, giữ vân nền), "darken" (làm tối) hoặc "lighten" (làm sáng)

        Returns:
            Ảnh đã thêm defect (copy, không thay đổi ảnh gốc)
        """
        result = image.copy()
        x, y, w, h = map(int, zone_rect)

        # Clamp vào image bounds
        img_h, img_w = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return result

        # Crop mask cho vùng hợp lệ
        mx1 = x1 - x
        my1 = y1 - y
        mx2 = mx1 + (x2 - x1)
        my2 = my1 + (y2 - y1)
        mask_crop = mask[my1:my2, mx1:mx2]

        # Alpha từ mask
        alpha = (mask_crop.astype(np.float32) / 255.0) * opacity

        if len(image.shape) == 3:
            alpha = np.expand_dims(alpha, axis=2)

        roi = result[y1:y2, x1:x2].astype(np.float32)

        if mode == "multiply":
            # Multiply blend: Giữ lại vân bề mặt bên dưới, vết lỗi tự nhiên ăn sâu vào chất liệu
            defect = roi * (1.0 - alpha * 0.95)
        elif mode == "darken":
            # Làm tối vùng defect
            defect = roi * (1.0 - alpha)
        elif mode == "lighten":
            # Làm sáng vùng defect
            defect = roi + (255.0 - roi) * alpha
        else:
            defect = roi * (1.0 - alpha)

        result[y1:y2, x1:x2] = np.clip(defect, 0, 255).astype(np.uint8)
        return result

    @classmethod
    def get_yolo_annotations(cls, bboxes, zone_rect, img_w, img_h, brush_type):
        """
        Chuyển bboxes thành YOLO format annotations.

        Args:
            bboxes: list of [x, y, w, h] tọa độ local trong zone
            zone_rect: [zx, zy, zw, zh] tọa độ zone trên ảnh gốc
            img_w, img_h: Kích thước ảnh gốc
            brush_type: str loại brush để map class_id

        Returns:
            list of str, mỗi dòng: "class_id center_x center_y width height"
        """
        annotations = []
        zx, zy = zone_rect[0], zone_rect[1]
        class_id = cls.CLASS_MAP.get(brush_type, 0)

        for bbox in bboxes:
            bx, by, bw, bh = bbox
            # Chuyển từ local zone coords sang global image coords
            gx = zx + bx
            gy = zy + by
            # YOLO format: center_x, center_y, width, height (normalized 0-1)
            cx = (gx + bw / 2.0) / img_w
            cy = (gy + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h
            # Clamp
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        return annotations

    # ==========================================
    # AUTO SURFACE DETECTION & SAFETY CLIPPING
    # ==========================================
    @staticmethod
    def detect_surfaces(img, target_surface='bright', threshold_mode='otsu',
                        manual_threshold=128, margin=4, min_area=150):
        """
        Tự động phân đoạn ảnh thành các vùng bề mặt Sáng (White) hoặc Tối (Dark)
        và co viền an toàn (Margin Erosion) để vết lỗi không bao giờ chạm ranh giới.

        Args:
            img: np.ndarray ảnh grayscale hoặc BGR
            target_surface: 'bright' (chỉ vùng sáng), 'dark' (chỉ vùng tối), hoặc 'both' (cả hai)
            threshold_mode: 'otsu' (tự động) hoặc 'manual' (theo manual_threshold)
            manual_threshold: int 0-255 nếu dùng 'manual'
            margin: int số pixel co viền an toàn (erosion) để tránh chạm mép
            min_area: int diện tích tối thiểu (pixel) của vùng bề mặt để nhận diện

        Returns:
            list of dict: [
                {
                    'rect': [x, y, w, h],           # Bounding box trên ảnh gốc
                    'contour': np.ndarray,          # Contour gốc của vùng
                    'surface_type': 'bright'|'dark',# Loại bề mặt
                    'defect_mode': 'multiply'|'lighten', # Mode hòa trộn đề xuất
                    'surface_mask': np.ndarray (h, w), # Binary mask cục bộ (đã margin)
                    'area': float
                }, ...
            ]
        """
        if img is None or img.size == 0:
            return []

        # Chuẩn hóa về Grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        h, w = img_gray.shape[:2]

        # Lọc nhẹ trước khi phân ngưỡng để loại bỏ nhiễu speckle
        blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Phân ngưỡng nhị phân
        if threshold_mode == 'otsu':
            _, thresh_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh_bin = (blur >= manual_threshold).astype(np.uint8) * 255

        # Danh sách các mặt cần phân tích
        surfaces_to_process = []
        if target_surface in ('bright', 'both'):
            surfaces_to_process.append(('bright', thresh_bin, 'multiply'))
        if target_surface in ('dark', 'both'):
            surfaces_to_process.append(('dark', cv2.bitwise_not(thresh_bin), 'lighten'))

        detected_zones = []

        # Kernel co viền an toàn (Margin Safety Erosion)
        kernel = None
        if margin > 0:
            k_size = margin * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        for surface_type, bin_mask, default_blend in surfaces_to_process:
            # 1. Đóng lỗ nhỏ (Morph Close) trước khi erode để mặt phẳng liền lạc
            clean_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            # 2. Áp dụng Margin Safety Erosion
            if kernel is not None:
                eroded_mask = cv2.erode(clean_mask, kernel)
            else:
                eroded_mask = clean_mask

            # 3. Tìm các vùng liên thông (Contours)
            cnts, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue

                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < 5 or bh < 5:
                    continue

                # Cắt mask cục bộ của vùng này đã lùi viền an toàn
                local_mask = eroded_mask[by:by+bh, bx:bx+bw].copy()

                detected_zones.append({
                    'rect': [bx, by, bw, bh],
                    'contour': c,
                    'surface_type': surface_type,
                    'defect_mode': default_blend,
                    'surface_mask': local_mask,
                    'area': area
                })

        return detected_zones
