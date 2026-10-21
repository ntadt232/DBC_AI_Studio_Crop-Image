"""
Worker Threads cho Defect & Dataset Studio AI.
Xử lý batch: PRS align → Apply brush defect → Save output + YOLO annotations (Detect, Seg, Pose) → Batch crop & Dataset Builder.
"""

import os
import glob
import random
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from utils import b64_to_img, img_to_b64, cv2_imread, cv2_imwrite, parse_yolo_line
from brush_engine import BrushEngine


# ==========================================
# THREAD XỬ LÝ CHÍNH: PRS + BRUSH DEFECT
# ==========================================
class DefectWorkerThread(QThread):
    """Luồng chạy nền: PRS matching → Affine align → Apply brush defects → Save output"""

    # Signals gửi về UI
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)          # current, total
    preview_signal = pyqtSignal(object, str)         # numpy image, filepath
    finished_signal = pyqtSignal(int, int)            # success_count, total_count

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

        # Load Template ảnh từ Base64 trong Recipe sang Numpy để match
        recipe = config.get('recipe', {})
        self.tpl1_img = b64_to_img(recipe.get('prs1', {}).get('tpl_b64', ''))
        self.tpl2_img = b64_to_img(recipe.get('prs2', {}).get('tpl_b64', ''))

    def find_pattern_in_search_area(self, img_gray, search_rect, tpl_img):
        """PRS: Tìm kiếm pattern trong vùng Search Area → trả về (score, global_point)"""
        sx, sy, sw, sh = search_rect
        h, w = img_gray.shape
        sx, sy = max(0, min(sx, w - 1)), max(0, min(sy, h - 1))
        sw, sh = max(1, min(sw, w - sx)), max(1, min(sh, h - sy))

        search_area = img_gray[sy:sy + sh, sx:sx + sw]

        # Bắt lỗi nếu vùng search bé hơn mẫu
        if search_area.shape[0] < tpl_img.shape[0] or search_area.shape[1] < tpl_img.shape[1]:
            return 0.0, (0, 0)

        res = cv2.matchTemplate(search_area, tpl_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Chuyển tọa độ cục bộ → tọa độ gốc
        global_pt = (sx + max_loc[0], sy + max_loc[1])
        return max_val, global_pt

    def align_image(self, img_gray):
        """PRS Align: template matching + affine transform"""
        recipe = self.config['recipe']
        prs1 = recipe.get('prs1', {})
        prs2 = recipe.get('prs2', {})
        prs1_thresh = prs1.get('threshold', 0.7)
        prs2_thresh = prs2.get('threshold', 0.7)

        # Kiểm tra recipe đã cấu hình PRS chưa
        if not prs1 or not prs2 or self.tpl1_img is None or self.tpl2_img is None:
            self.log_signal.emit("  ⚠ Recipe chưa cấu hình PRS đầy đủ — sử dụng ảnh gốc không align")
            return img_gray, True  # Trả về ảnh gốc, vẫn tiếp tục

        # PRS Matching
        val1, pt1 = self.find_pattern_in_search_area(img_gray, prs1['search_rect'], self.tpl1_img)
        val2, pt2 = self.find_pattern_in_search_area(img_gray, prs2['search_rect'], self.tpl2_img)

        self.log_signal.emit(f"  PRS1: {val1:.4f} (≥{prs1_thresh})  |  PRS2: {val2:.4f} (≥{prs2_thresh})")

        if val1 < prs1_thresh or val2 < prs2_thresh:
            self.log_signal.emit(f"  ❌ PRS Failed — bỏ qua ảnh này")
            return None, False

        # Affine Transform
        src_pts = np.float32([pt1, pt2])
        dst_pts = np.float32([prs1['ref_pt'], prs2['ref_pt']])
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if matrix is None:
            self.log_signal.emit(f"  ❌ Không thể tính Affine Transform — bỏ qua")
            return None, False

        h, w = img_gray.shape
        aligned_img = cv2.warpAffine(img_gray, matrix, (w, h))
        return aligned_img, True

    def process_image(self, filepath):
        """
        Xử lý 1 file ảnh: Load → (PRS Align hoặc Auto Surface) → Apply Brush Defects → Save.
        Returns: True nếu thành công
        """
        try:
            # 1. Load ảnh
            img = cv2_imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.log_signal.emit(f"  ⚠ Không đọc được ảnh: {os.path.basename(filepath)}")
                return False

            surface_params = self.config.get('surface_params', {})
            auto_surface_enabled = surface_params.get('enabled', False)

            if auto_surface_enabled:
                aligned_img = img.copy()
            else:
                # 2. PRS Align
                aligned_img, success = self.align_image(img)
                if not success or aligned_img is None:
                    return False

            # 3. Apply Brush Defects
            brush_params = self.config.get('brush_params', BrushEngine.default_params())
            brush_type = self.config.get('brush_type', 'random')
            defect_mode = self.config.get('defect_mode', 'darken')

            result_img = aligned_img.copy()
            all_annotations = []
            img_h, img_w = result_img.shape[:2]
            defect_summary = []

            if auto_surface_enabled:
                # CHẾ ĐỘ AUTO SURFACE DETECTION
                detected_zones = BrushEngine.detect_surfaces(
                    result_img,
                    target_surface=surface_params.get('target_surface', 'bright'),
                    threshold_mode=surface_params.get('threshold_mode', 'otsu'),
                    manual_threshold=surface_params.get('manual_threshold', 128),
                    margin=surface_params.get('margin', 4),
                    min_area=surface_params.get('min_area', 200)
                )

                if not detected_zones:
                    self.log_signal.emit(f"  ⚠ Không tìm thấy vùng bề mặt phù hợp trên {os.path.basename(filepath)}")
                    return False

                zone_counts = BrushEngine.allocate_counts([z['area'] for z in detected_zones], brush_params)
                for z_idx, zone in enumerate(detected_zones):
                    if zone_counts[z_idx] == 0:
                        continue
                    zone_params = dict(brush_params, count_min=zone_counts[z_idx], count_max=zone_counts[z_idx])
                    if not self.is_running:
                        return False

                    zx, zy, zw, zh = zone['rect']
                    if zw < 5 or zh < 5:
                        continue

                    # Tạo defect, sinh lại tối đa 15 lần cho tới khi nằm trên bề mặt (đã co viền an toàn)
                    best = None
                    for _try in range(15):
                        raw_mask, bboxes, actual_type = BrushEngine.generate_defect(
                            zw, zh, brush_type, zone_params)
                        raw_px = np.count_nonzero(raw_mask)
                        clipped = cv2.bitwise_and(raw_mask, zone['surface_mask'])
                        ratio = np.count_nonzero(clipped) / raw_px if raw_px else 0.0
                        if best is None or ratio > best[0]:
                            best = (ratio, clipped, bboxes, actual_type)
                        if ratio >= 0.95:  # vết nằm trọn trong bề mặt -> không bị cắt vụn
                            break
                    _, mask, bboxes, actual_type = best

                    # Tính lại bboxes
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0]

                    if np.sum(mask) == 0 or not bboxes:
                        continue

                    # Apply defect
                    opacity = random.uniform(brush_params.get('opacity_min', 0.3), brush_params.get('opacity_max', 0.9))
                    zone_mode = zone['defect_mode'] if surface_params.get('target_surface') == 'both' else defect_mode
                    result_img = BrushEngine.apply_defect(
                        result_img, [zx, zy, zw, zh], mask, opacity, zone_mode)

                    # YOLO Annotation
                    class_id = BrushEngine.CLASS_MAP.get(actual_type, 0)
                    yolo_format = self.config.get('yolo_format', 'Bounding Box')

                    if "Segmentation" in yolo_format:
                        for c in contours:
                            if cv2.contourArea(c) < 5:  # Vết quá nhỏ -> dùng khung bao làm polygon thay vì bỏ nhãn
                                _bx, _by, _bw, _bh = cv2.boundingRect(c)
                                _bw, _bh = max(_bw, 1), max(_bh, 1)
                                c = np.array([[[_bx, _by]], [[_bx + _bw, _by]], [[_bx + _bw, _by + _bh]], [[_bx, _by + _bh]]], dtype=np.int32)
                            epsilon = 0.002 * cv2.arcLength(c, True)
                            c = cv2.approxPolyDP(c, epsilon, True)
                            c = c.squeeze(1) if len(c.shape) == 3 else c
                            points_str = []
                            for pt in c:
                                abs_x = zx + pt[0]
                                abs_y = zy + pt[1]
                                norm_x = max(0.0, min(1.0, abs_x / img_w))
                                norm_y = max(0.0, min(1.0, abs_y / img_h))
                                points_str.append(f"{norm_x:.6f} {norm_y:.6f}")
                            if len(points_str) >= 3:
                                all_annotations.append(f"{class_id} " + " ".join(points_str))
                    else:
                        for bx, by, bw, bh in bboxes:
                            abs_x = zx + bx
                            abs_y = zy + by
                            x_center = (abs_x + bw / 2) / img_w
                            y_center = (abs_y + bh / 2) / img_h
                            w_norm = bw / img_w
                            h_norm = bh / img_h
                            all_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                    defect_summary.append(f"Surf_{z_idx+1}:{actual_type}")
                    self.log_signal.emit(f"  ✅ Vùng {zone['surface_type']} #{z_idx+1}: {actual_type} (opacity={opacity:.2f}, bboxes={len(bboxes)})")

            else:
                # CHẾ ĐỘ THƯỜNG DỰA TRÊN RECIPE DEFECT ZONES
                recipe = self.config.get('recipe', {})
                zones = recipe.get('defect_zones', [])

                zone_counts = BrushEngine.allocate_counts([(cv2.contourArea(np.array(z['points'], dtype=np.int32)) if z.get('type') == 'polygon' else float(z['rect'][2]) * float(z['rect'][3])) for z in zones], brush_params)
                for z_idx, zone in enumerate(zones):
                    if zone_counts[z_idx] == 0:
                        continue
                    zone_params = dict(brush_params, count_min=zone_counts[z_idx], count_max=zone_counts[z_idx])
                    if not self.is_running:
                        return False

                    zone_name = zone.get('name', f'Zone_{z_idx + 1}')

                    if zone.get("type") == "polygon":
                        pts = np.array(zone["points"], dtype=np.int32)
                        zx, zy, zw, zh = cv2.boundingRect(pts)
                    else:
                        zone_rect = zone['rect'] if isinstance(zone, dict) else zone
                        zx, zy, zw, zh = map(int, zone_rect)

                    # Clamp vào ảnh
                    zx = max(0, min(zx, img_w - 1))
                    zy = max(0, min(zy, img_h - 1))
                    zw = max(1, min(zw, img_w - zx))
                    zh = max(1, min(zh, img_h - zy))

                    if zw < 5 or zh < 5:
                        self.log_signal.emit(f"  ⚠ {zone_name}: Zone quá nhỏ, bỏ qua")
                        continue

                    # Tạo defect
                    mask, bboxes, actual_type = BrushEngine.generate_defect(
                        zw, zh, brush_type, zone_params)

                    # Nếu là polygon, cắt mask theo hình đa giác
                    if zone.get("type") == "polygon":
                        local_pts = pts - np.array([zx, zy])
                        poly_mask = np.zeros((zh, zw), dtype=np.uint8)
                        cv2.fillPoly(poly_mask, [local_pts], 255)
                        mask = cv2.bitwise_and(mask, poly_mask)
                        # Tính lại bboxes
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0]

                    if np.sum(mask) == 0 or not bboxes:
                        continue

                    # Apply
                    opacity = random.uniform(brush_params.get('opacity_min', 0.3), brush_params.get('opacity_max', 0.9))
                    result_img = BrushEngine.apply_defect(
                        result_img, [zx, zy, zw, zh], mask, opacity, defect_mode)

                    # YOLO Annotation
                    class_id = BrushEngine.CLASS_MAP.get(actual_type, 0)
                    yolo_format = self.config.get('yolo_format', 'Bounding Box')

                    if "Segmentation" in yolo_format:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in contours:
                            if cv2.contourArea(c) < 5:  # Vết quá nhỏ -> dùng khung bao làm polygon thay vì bỏ nhãn
                                _bx, _by, _bw, _bh = cv2.boundingRect(c)
                                _bw, _bh = max(_bw, 1), max(_bh, 1)
                                c = np.array([[[_bx, _by]], [[_bx + _bw, _by]], [[_bx + _bw, _by + _bh]], [[_bx, _by + _bh]]], dtype=np.int32)
                            
                            epsilon = 0.002 * cv2.arcLength(c, True)
                            c = cv2.approxPolyDP(c, epsilon, True)
                            c = c.squeeze(1) if len(c.shape) == 3 else c
                            
                            points_str = []
                            for pt in c:
                                abs_x = zx + pt[0]
                                abs_y = zy + pt[1]
                                norm_x = max(0.0, min(1.0, abs_x / img_w))
                                norm_y = max(0.0, min(1.0, abs_y / img_h))
                                points_str.append(f"{norm_x:.6f} {norm_y:.6f}")
                                
                            if len(points_str) >= 3:
                                all_annotations.append(f"{class_id} " + " ".join(points_str))
                    else:
                        for bx, by, bw, bh in bboxes:
                            abs_x = zx + bx
                            abs_y = zy + by
                            x_center = (abs_x + bw / 2) / img_w
                            y_center = (abs_y + bh / 2) / img_h
                            w_norm = bw / img_w
                            h_norm = bh / img_h
                            all_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                    defect_summary.append(f"{zone_name}:{actual_type}")
                    self.log_signal.emit(f"  ✅ {zone_name}: {actual_type} (opacity={opacity:.2f}, bboxes={len(bboxes)})")

            # 4. Lưu ảnh output
            output_folder = self.config.get('output_folder', '')
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                ext = self.config.get('output_format', '.png')

                output_path = os.path.join(output_folder, f"{base_name}_{brush_type}{ext}")
                cv2_imwrite(output_path, result_img)

                # 5. Lưu YOLO annotations
                if self.config.get('yolo_export', False) and all_annotations:
                    labels_folder = self.config.get('labels_folder', '')
                    if not labels_folder:
                        labels_folder = os.path.join(output_folder, 'labels')
                    os.makedirs(labels_folder, exist_ok=True)
                    label_path = os.path.join(labels_folder, f"{base_name}_{brush_type}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(all_annotations))

            # Emit preview
            summary_str = ", ".join(defect_summary)
            self.preview_signal.emit(result_img, filepath)
            self.log_signal.emit(f"  💾 Đã lưu: {os.path.basename(filepath)} → [{summary_str}]")
            return True

        except Exception as e:
            self.log_signal.emit(f"  ❌ Lỗi xử lý {os.path.basename(filepath)}: {e}")
            return False

    def run(self):
        """Entry point — chạy batch xử lý tuần tự"""
        files = self.config.get('input_files', [])
        total = len(files)
        success_count = 0

        brush_types = self.config.get('brush_types', ['random'])
        if not isinstance(brush_types, list):
            brush_types = [brush_types]

        self.log_signal.emit(f"═══ Bắt đầu tạo Defect: {total} ảnh ═══")
        self.log_signal.emit(f"  Brushes: {', '.join(brush_types)}")
        surf_params = self.config.get('surface_params', {})
        if surf_params.get('enabled', False):
            self.log_signal.emit(f"  Chế độ: 🤖 Auto Surface Detection ({surf_params.get('target_surface')}, margin={surf_params.get('margin')}px)")
        else:
            self.log_signal.emit(f"  Zones: {len(self.config.get('recipe', {}).get('defect_zones', []))}")
        self.log_signal.emit(f"  YOLO Export: {'Bật' if self.config.get('yolo_export', False) else 'Tắt'}")
        self.log_signal.emit("")

        for i, filepath in enumerate(files):
            if not self.is_running:
                self.log_signal.emit("⏹ Đã dừng bởi người dùng.")
                break

            self.log_signal.emit(f"[{i + 1}/{total}] 📷 {os.path.basename(filepath)}")
            self.progress_signal.emit(i + 1, total)

            for b_type in brush_types:
                if not self.is_running:
                    break
                
                # Cập nhật brush_type hiện tại cho process_image
                self.config['brush_type'] = b_type
                if self.process_image(filepath):
                    success_count += 1

        total_jobs = total * len(brush_types)
        self.log_signal.emit(f"\n═══ Hoàn tất: {success_count}/{total_jobs} ảnh output thành công ({total} ảnh gốc × {len(brush_types)} brush) ═══")

        # Tạo classes.txt nếu YOLO export
        if self.config.get('yolo_export', False):
            labels_folder = self.config.get('labels_folder', '')
            if not labels_folder:
                labels_folder = os.path.join(self.config.get('output_folder', ''), 'labels')
            if os.path.isdir(labels_folder):
                classes_path = os.path.join(labels_folder, 'classes.txt')
                with open(classes_path, 'w') as f:
                    for btype in BrushEngine.ALL_TYPES:
                        f.write(f"{btype}\n")
                self.log_signal.emit(f"📄 Đã tạo classes.txt: {classes_path}")

        self.finished_signal.emit(success_count, total_jobs)

    def stop(self):
        """Dừng thread an toàn"""
        self.is_running = False



def find_outside_in_segmentation(patch_bin, dzone_poly, ignore_polies=None, center=None, n_rays=360, epsilon_ratio=0.005, polarity='black_to_white', search_direction='outside_in', dzone_fallback=True, search_dist=100):
    """
    Dò đường biên từ ngoài vào trong (hoặc từ trong ra ngoài):
    - Quét từ đường biên Defect Zone (ngoài) hướng về tâm (trong), hoặc ngược lại.
    - search_dist: Khoảng cách quét tối đa tính từ điểm bắt đầu (ví dụ 100 px).
      Nếu trong khoảng cách này không tìm thấy chuyển màu -> tự động lấy đường biên Defect Zone (px, py).
    - Nếu đụng đường viền của Ignore Zone -> lấy ngay đường viền tại vị trí đụng.
    - Nếu gặp chuyển mức trên ảnh nhị phân (Đen ↔ Trắng) -> lấy điểm chuyển mức.
    - Nếu tia quét không có chuyển mức:
      + dzone_fallback=True: tự nhận đường biên của Defect Zone làm điểm segment.
      + dzone_fallback=False: bỏ qua điểm đó.
    """
    H, W = patch_bin.shape[:2]
    
    ignore_mask = np.zeros((H, W), dtype=np.uint8)
    if ignore_polies:
        for ipoly in ignore_polies:
            if len(ipoly) >= 3:
                cv2.fillPoly(ignore_mask, [ipoly.astype(np.int32)], 255)
                
    V = dzone_poly.astype(np.float32)
    M = len(V)
    if center is None:
        cx = float(np.mean(V[:, 0]))
        cy = float(np.mean(V[:, 1]))
    else:
        cx, cy = float(center[0]), float(center[1])
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_bin = cv2.morphologyEx(patch_bin, cv2.MORPH_OPEN, kernel)
    
    edges = np.zeros((M, 2), dtype=np.float32)
    for j in range(M):
        edges[j] = V[(j + 1) % M] - V[j]
        
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    pts = []
    
    for i in range(n_rays):
        dx, dy = cos_a[i], sin_a[i]
        
        # Giao điểm của tia (cx, cy) + t*(dx, dy) với các cạnh của đa giác Defect Zone
        best_t = 1e9
        for j in range(M):
            ex, ey = edges[j, 0], edges[j, 1]
            denom = dx * ey - dy * ex
            if abs(denom) > 1e-6:
                vx = V[j, 0] - cx
                vy = V[j, 1] - cy
                t_cand = (vx * ey - vy * ex) / denom
                u_cand = (vx * dy - vy * dx) / denom
                if t_cand > 0.1 and 0.0 <= u_cand <= 1.0:
                    if t_cand < best_t:
                        best_t = t_cand
                        
        if best_t >= 1e8:
            best_t = 50.0
            
        px = max(0.0, min(float(W - 1), cx + best_t * dx))
        py = max(0.0, min(float(H - 1), cy + best_t * dy))
        
        dist = int(np.hypot(px - cx, py - cy))
        if dist <= 1:
            if dzone_fallback:
                pts.append((px, py))
            continue
            
        # Xác định hướng quét
        if search_direction == 'inside_out':
            # Quét từ tâm ra ngoài
            start_x, start_y = cx, cy
            end_x, end_y = px, py
        else:
            # Quét từ ngoài vào trong (default)
            start_x, start_y = px, py
            end_x, end_y = cx, cy
        
        # Khoảng cách quét tối đa theo search_dist (nếu > 0)
        max_scan = min(dist, search_dist) if search_dist > 0 else dist
        if max_scan < 1:
            max_scan = 1
        
        prev_ix = int(round(start_x))
        prev_iy = int(round(start_y))
        prev_ix = max(0, min(W - 1, prev_ix))
        prev_iy = max(0, min(H - 1, prev_iy))
        prev_in_ig = ignore_mask[prev_iy, prev_ix] > 0
        prev_val = int(cleaned_bin[prev_iy, prev_ix])
        
        # Kiểm tra xem điểm bắt đầu có nằm sẵn trong đối tượng từ trước hay không:
        # Nếu dò Ngoài → Trong (Outside-In):
        # - polarity 'white_to_black': đối tượng là Đen (0). Nếu điểm bắt đầu tại biên Defect Zone đã là Đen
        # - polarity 'black_to_white': đối tượng là Trắng (255). Nếu điểm bắt đầu tại biên Defect Zone đã là Trắng
        # -> Đối tượng đã tràn ra mép Defect Zone từ trước. Không được bắt các chuyển mức bên trong
        #    (vì sẽ bắt nhầm đốm sáng phản quang hoặc bóng đổ bên trong thân đối tượng gây răng cưa nhọn).
        started_in_target = False
        if search_direction == 'outside_in' and not prev_in_ig:
            if polarity == 'white_to_black':
                start_samples = [prev_val]
                for check_s in range(1, min(4, dist + 1)):
                    c_alpha = check_s / dist
                    c_ix = int(round(start_x + c_alpha * (end_x - start_x)))
                    c_iy = int(round(start_y + c_alpha * (end_y - start_y)))
                    if 0 <= c_ix < W and 0 <= c_iy < H:
                        start_samples.append(int(cleaned_bin[c_iy, c_ix]))
                if start_samples.count(0) >= (len(start_samples) + 1) // 2:
                    started_in_target = True
            elif polarity == 'black_to_white':
                start_samples = [prev_val]
                for check_s in range(1, min(4, dist + 1)):
                    c_alpha = check_s / dist
                    c_ix = int(round(start_x + c_alpha * (end_x - start_x)))
                    c_iy = int(round(start_y + c_alpha * (end_y - start_y)))
                    if 0 <= c_ix < W and 0 <= c_iy < H:
                        start_samples.append(int(cleaned_bin[c_iy, c_ix]))
                if start_samples.count(255) >= (len(start_samples) + 1) // 2:
                    started_in_target = True

        found_edge = False
        has_color_change = False
        chosen = None
        
        for s in range(1, max_scan + 1):
            alpha = s / dist
            qx = start_x + alpha * (end_x - start_x)
            qy = start_y + alpha * (end_y - start_y)
            ix = int(round(qx))
            iy = int(round(qy))
            if not (0 <= ix < W and 0 <= iy < H):
                continue
                
            in_ig = ignore_mask[iy, ix] > 0
            val = int(cleaned_bin[iy, ix])
            
            if val != prev_val:
                has_color_change = True
                
            # 1. Đụng đường viền Ignore Zone (chuyển trạng thái vào/ra khỏi ignore zone):
            if prev_in_ig != in_ig:
                chosen = (qx, qy)
                found_edge = True
                break
                
            # 2. Chuyển mức màu nhị phân khi nằm ngoài Ignore Zone:
            # Chỉ bắt chuyển mức khi tia KHÔNG bắt đầu từ bên trong đối tượng từ trước
            if not in_ig and not started_in_target:
                if polarity == 'black_to_white':
                    if prev_val == 0 and val == 255:
                        chosen = (qx, qy)
                        found_edge = True
                        break
                elif polarity == 'white_to_black':
                    if prev_val == 255 and val == 0:
                        chosen = (qx, qy)
                        found_edge = True
                        break
                else:  # any
                    if prev_val != val:
                        chosen = (qx, qy)
                        found_edge = True
                        break
                        
            prev_in_ig = in_ig
            prev_val = val
            
        if found_edge:
            pts.append(chosen)
        else:
            # Không tìm thấy điểm biên thỏa mãn polarity hoặc bắt đầu trong đối tượng
            if dzone_fallback or started_in_target:
                # Tự động lấy đường biên của Defect Zone làm điểm segment
                pts.append((px, py))
        
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float32)
        
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(pts_arr, True)
    eps = max(1.0, epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(pts_arr, eps, True)
    return approx.reshape(-1, 2)


def compute_pose_for_roi(roi, img_gray, img_w, img_h, class_id, pose_kpt_mode, num_kpts, margin, config, M=None):
    """
    Tính toán Bounding Box và tự động chấm Keypoints cho 1 vùng Defect Zone đã biến đổi theo PRS.
    Bounding Box chính là vùng Defect Zone.
    Keypoints lấy trực tiếp từ điểm chấm Tab 1 biến đổi qua ma trận PRS (M),
    hoặc tự động tính theo hình học/PCA nếu chưa có điểm Master.
    Returns: (annot_line, preview_dict)
    """
    if roi.get('poly_pts') is not None:
        pts = roi['poly_pts']
    elif roi.get('raw_corners') is not None:
        pts = roi['raw_corners']
    else:
        bx, by, bw, bh = roi['bbox']
        pts = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.int32)

    bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    if margin > 0:
        bx = max(0, bx - margin)
        by = max(0, by - margin)
        bw = min(img_w - bx, bw + 2 * margin)
        bh = min(img_h - by, bh + 2 * margin)
    else:
        bx = max(0, min(bx, img_w - 1))
        by = max(0, min(by, img_h - 1))
        bw = max(1, min(bw, img_w - bx))
        bh = max(1, min(bh, img_h - by))

    center_x = float(np.mean(pts[:, 0]))
    center_y = float(np.mean(pts[:, 1]))

    keypoints = []

    # Ưu tiên 1: Lấy điểm chấm Master từ Tab 1 và biến đổi qua ma trận PRS (M)
    master_kpts = roi.get('master_kpts')
    if master_kpts:
        pts_to_trans = []
        visibilities = []
        for kp in master_kpts:
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                pts_to_trans.append([float(kp[0]), float(kp[1])])
                visibilities.append(int(kp[2]) if len(kp) > 2 else 2)
            elif isinstance(kp, dict):
                pos = kp.get('pos', [0, 0])
                pts_to_trans.append([float(pos[0]), float(pos[1])])
                visibilities.append(int(kp.get('v', 2)))
        if pts_to_trans:
            pts_arr = np.array([pts_to_trans], dtype=np.float32)
            if M is not None:
                trans_pts = cv2.transform(pts_arr, M)[0]
            else:
                trans_pts = pts_arr[0]
            for (tx, ty), v in zip(trans_pts, visibilities):
                keypoints.append((float(tx), float(ty), v))

    if not keypoints:
        if pose_kpt_mode == "center" or num_kpts == 1:
            keypoints.append((center_x, center_y, 2))
        elif pose_kpt_mode == "corners" or num_kpts == 4:
            if len(pts) == 4:
                for p in pts:
                    keypoints.append((float(p[0]), float(p[1]), 2))
            else:
                keypoints = [
                    (float(bx), float(by), 2),
                    (float(bx + bw), float(by), 2),
                    (float(bx + bw), float(by + bh), 2),
                    (float(bx), float(by + bh), 2)
                ]
        elif pose_kpt_mode == "centroid_blob":
            patch = img_gray[by:by + bh, bx:bx + bw]
            blob_found = False
            if patch.size > 0:
                blurred = cv2.GaussianBlur(patch, (3, 3), 0)
                th_val = config.get('threshold', 128)
                use_otsu = config.get('use_otsu', False)
                thresh_mode = cv2.THRESH_BINARY
                if use_otsu:
                    _, bin_patch = cv2.threshold(blurred, 0, 255, thresh_mode | cv2.THRESH_OTSU)
                else:
                    _, bin_patch = cv2.threshold(blurred, th_val, 255, thresh_mode)

                cnts, _ = cv2.findContours(bin_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    largest_cnt = max(cnts, key=cv2.contourArea)
                    M_mom = cv2.moments(largest_cnt)
                    if M_mom["m00"] > 0:
                        blob_cx = bx + M_mom["m10"] / M_mom["m00"]
                        blob_cy = by + M_mom["m01"] / M_mom["m00"]
                        blob_found = True
                        if num_kpts == 1:
                            keypoints.append((float(blob_cx), float(blob_cy), 2))
                        elif num_kpts == 2:
                            cnt_pts = largest_cnt.reshape(-1, 2).astype(np.float32)
                            if len(cnt_pts) >= 2:
                                try:
                                    mean_c, eigen_c = cv2.PCACompute(cnt_pts, mean=None)
                                    dir_v = eigen_c[0]
                                    proj_c = np.dot(cnt_pts - mean_c, dir_v.T).squeeze()
                                    p_min = float(np.min(proj_c))
                                    p_max = float(np.max(proj_c))
                                    k1 = mean_c[0] + dir_v * p_min + np.array([bx, by])
                                    k2 = mean_c[0] + dir_v * p_max + np.array([bx, by])
                                    keypoints.append((float(k1[0]), float(k1[1]), 2))
                                    keypoints.append((float(k2[0]), float(k2[1]), 2))
                                except Exception:
                                    keypoints.append((float(blob_cx - bw * 0.25), float(blob_cy), 2))
                                    keypoints.append((float(blob_cx + bw * 0.25), float(blob_cy), 2))
                            else:
                                keypoints.append((float(blob_cx - bw * 0.25), float(blob_cy), 2))
                                keypoints.append((float(blob_cx + bw * 0.25), float(blob_cy), 2))
                        else:
                            keypoints.append((float(blob_cx), float(blob_cy), 2))
            if not blob_found:
                keypoints.append((center_x, center_y, 2))
        else:
            # 'two_points', 'equidistant', hoặc tổng quát N điểm theo trục chính
            pts_f = pts.astype(np.float32)
            if len(pts_f) >= 2:
                try:
                    mean, eigenvectors = cv2.PCACompute(pts_f, mean=None)
                    dir_vec = eigenvectors[0]
                    proj = np.dot(pts_f - mean, dir_vec.T).squeeze()
                    p_min = float(np.min(proj))
                    p_max = float(np.max(proj))
                    if num_kpts == 1:
                        keypoints.append((float(mean[0][0]), float(mean[0][1]), 2))
                    elif num_kpts == 2:
                        kpt1 = mean[0] + dir_vec * (p_min * 0.85)
                        kpt2 = mean[0] + dir_vec * (p_max * 0.85)
                        keypoints.append((float(kpt1[0]), float(kpt1[1]), 2))
                        keypoints.append((float(kpt2[0]), float(kpt2[1]), 2))
                    else:
                        t_vals = np.linspace(p_min * 0.85, p_max * 0.85, num_kpts)
                        for t in t_vals:
                            kp = mean[0] + dir_vec * t
                            keypoints.append((float(kp[0]), float(kp[1]), 2))
                except Exception:
                    if bw >= bh:
                        k1 = (bx + 0.15 * bw, by + 0.5 * bh)
                        k2 = (bx + 0.85 * bw, by + 0.5 * bh)
                    else:
                        k1 = (bx + 0.5 * bw, by + 0.15 * bh)
                        k2 = (bx + 0.5 * bw, by + 0.85 * bh)
                    keypoints.append((float(k1[0]), float(k1[1]), 2))
                    keypoints.append((float(k2[0]), float(k2[1]), 2))
            else:
                keypoints.append((center_x, center_y, 2))

    while len(keypoints) < num_kpts:
        keypoints.append((center_x, center_y, 2))
    if len(keypoints) > num_kpts:
        keypoints = keypoints[:num_kpts]

    clamped_kpts = []
    for kx, ky, v in keypoints:
        if 0 <= kx < img_w and 0 <= ky < img_h:
            clamped_kpts.append((float(kx), float(ky), int(v)))
        else:
            cx_c = max(0.0, min(float(img_w - 1), float(kx)))
            cy_c = max(0.0, min(float(img_h - 1), float(ky)))
            clamped_kpts.append((cx_c, cy_c, 0))

    x_c = (bx + bw / 2.0) / float(img_w)
    y_c = (by + bh / 2.0) / float(img_h)
    w_n = float(bw) / float(img_w)
    h_n = float(bh) / float(img_h)
    kpt_tokens = [f"{kx / float(img_w):.6f} {ky / float(img_h):.6f} {kv}" for kx, ky, kv in clamped_kpts]
    annot_line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f} " + " ".join(kpt_tokens)

    preview_item = {
        'bbox': (bx, by, bw, bh),
        'keypoints': clamped_kpts,
        'class_id': class_id
    }
    return annot_line, preview_item


# ==========================================
# THREAD XỬ LÝ BATCH: TÌM PRS → ĐỊNH VỊ DEFECT ZONE → OUTSIDE-IN SEGMENTATION → YOLO-SEG / YOLO-POSE
# ==========================================
class AutoPrsSegWorkerThread(QThread):
    """Luồng chạy nền: 
      1. Dò PRS (Pattern 1 & Pattern 2 nếu có) từ Tab 1 để tính độ dịch chuyển/biến đổi so với Master.
      2. Tịnh tiến / biến đổi các vùng Defect Zone từ Tab 1 sang ảnh mục tiêu.
         (Nếu chưa vẽ Defect Zone, dùng trực tiếp vị trí Pattern 1 tìm được).
      3. Phân vùng nhị phân (Binary Threshold) bên trong vùng Defect Zone.
      4. CHỈ CHỌN 1 ĐỐI TƯỢNG DUY NHẤT có diện tích lớn nhất (loại bỏ toàn bộ đối tượng nhỏ hơn là vẽ nhầm).
      5. Xuất nhãn YOLO-Seg polygon chuẩn.
    """

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)          # current, total
    preview_signal = pyqtSignal(object, str, list)   # numpy image, filepath, list of polies
    finished_signal = pyqtSignal(int, int)          # success_count, total_count

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

        recipe = config.get('recipe', {})
        self.tpl1_img = b64_to_img(recipe.get('prs1', {}).get('tpl_b64', ''))
        self.tpl2_img = b64_to_img(recipe.get('prs2', {}).get('tpl_b64', ''))

    def find_best_match(self, img_gray, search_rect, tpl_img):
        """Tìm vị trí khớp tốt nhất của pattern trong search area"""
        if tpl_img is None or tpl_img.size == 0:
            return 0.0, None
        h, w = img_gray.shape[:2]
        if search_rect:
            sx, sy, sw, sh = search_rect
            sx, sy = max(0, min(sx, w - 1)), max(0, min(sy, h - 1))
            sw = max(1, min(sw, w - sx))
            sh = max(1, min(sh, h - sy))
        else:
            sx, sy, sw, sh = 0, 0, w, h

        search_area = img_gray[sy:sy + sh, sx:sx + sw]
        tpl_h, tpl_w = tpl_img.shape[:2]
        if search_area.shape[0] < tpl_h or search_area.shape[1] < tpl_w:
            return 0.0, None

        res = cv2.matchTemplate(search_area, tpl_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        global_pt = (sx + max_loc[0], sy + max_loc[1])
        return float(max_val), global_pt

    def process_single_image(self, filepath):
        """
        Dò PRS và tạo đường bao cho đúng 1 đối tượng duy nhất trong Defect Zone.
        Returns: (success, result_img, annotations_list, polies_for_preview)
        """
        img_bgr = cv2_imread(filepath)
        if img_bgr is None:
            self.log_signal.emit(f"  ⚠ Không đọc được file ảnh: {os.path.basename(filepath)}")
            return False, None, [], []

        if len(img_bgr.shape) == 3:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_bgr
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        h, w = img_gray.shape[:2]
        recipe = self.config.get('recipe', {})
        prs1 = recipe.get('prs1', {})
        prs2 = recipe.get('prs2', {})
        th1 = prs1.get('threshold', 0.6)
        th2 = prs2.get('threshold', 0.6)

        # 1. Tìm PRS (Pattern 1 & Pattern 2)
        has_prs1 = self.tpl1_img is not None and self.tpl1_img.size > 0
        has_prs2 = self.tpl2_img is not None and self.tpl2_img.size > 0

        val1, pt1 = 0.0, None
        val2, pt2 = 0.0, None
        if has_prs1:
            val1, pt1 = self.find_best_match(img_gray, prs1.get('search_rect'), self.tpl1_img)
            self.log_signal.emit(f"  PRS 1 Match Score: {val1:.3f} (Ngưỡng: {th1:.2f})")
        if has_prs2:
            val2, pt2 = self.find_best_match(img_gray, prs2.get('search_rect'), self.tpl2_img)
            self.log_signal.emit(f"  PRS 2 Match Score: {val2:.3f} (Ngưỡng: {th2:.2f})")

        # Tính ma trận dịch chuyển / biến đổi M từ Master sang Target
        M = None
        if has_prs1 and has_prs2 and val1 >= th1 and val2 >= th2:
            src_pts = np.float32([prs1.get('ref_pt', pt1), prs2.get('ref_pt', pt2)])
            dst_pts = np.float32([pt1, pt2])
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if M is not None:
                self.log_signal.emit("  ✅ Đã khớp cả 2 PRS (Tính biến đổi Affine)")
            else:
                self.log_signal.emit("  ⚠ Affine thất bại, chuyển sang tịnh tiến theo PRS 1")
                ref_pt1 = prs1.get('ref_pt', pt1)
                dx = float(pt1[0] - ref_pt1[0])
                dy = float(pt1[1] - ref_pt1[1])
                M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        elif has_prs1 and val1 >= th1 and pt1 is not None:
            ref_pt1 = prs1.get('ref_pt', pt1)
            dx = float(pt1[0] - ref_pt1[0])
            dy = float(pt1[1] - ref_pt1[1])
            M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            self.log_signal.emit(f"  ✅ Đã khớp PRS 1 (Độ dịch chuyển dx={int(dx)}, dy={int(dy)})")
        else:
            if has_prs1 and val1 < th1:
                self.log_signal.emit(f"  ⚠ PRS 1 score {val1:.3f} < ngưỡng {th1:.2f} (Giữ nguyên tọa độ gốc)")
            M = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # 2. Định vị các vùng Defect Zone dựa trên PRS
        defect_zones = recipe.get('defect_zones', [])
        target_rois = []

        if defect_zones:
            for zone in defect_zones:
                master_kpts = zone.get('keypoints')
                if zone.get("type") == "polygon":
                    pts = np.array(zone.get("points", []), dtype=np.float32)
                    if len(pts) >= 3:
                        pts_trans = cv2.transform(np.array([pts]), M)[0].astype(np.int32)
                        bx, by, bw, bh = cv2.boundingRect(pts_trans)
                        target_rois.append({
                            'bbox': (bx, by, bw, bh),
                            'poly_pts': pts_trans,
                            'master_kpts': master_kpts
                        })
                else:
                    rect = zone.get('rect', [0, 0, w, h])
                    rx, ry, rw_z, rh_z = rect[0], rect[1], rect[2], rect[3]
                    corners = np.float32([[rx, ry], [rx + rw_z, ry], [rx + rw_z, ry + rh_z], [rx, ry + rh_z]])
                    corners_trans = cv2.transform(np.array([corners]), M)[0].astype(np.int32)
                    bx, by, bw, bh = cv2.boundingRect(corners_trans)
                    target_rois.append({
                        'bbox': (bx, by, bw, bh),
                        'poly_pts': None,
                        'raw_corners': corners_trans,
                        'master_kpts': master_kpts
                    })
        elif pt1 is not None and has_prs1:
            # Nếu chưa vẽ Defect Zone, lấy chính vùng Pattern 1 làm ROI
            tpl_h, tpl_w = self.tpl1_img.shape[:2]
            p_corners = np.array([[pt1[0], pt1[1]], [pt1[0] + tpl_w, pt1[1]], [pt1[0] + tpl_w, pt1[1] + tpl_h], [pt1[0], pt1[1] + tpl_h]], dtype=np.int32)
            target_rois.append({
                'bbox': (pt1[0], pt1[1], tpl_w, tpl_h),
                'poly_pts': None,
                'raw_corners': p_corners,
                'master_kpts': None
            })
        else:
            target_rois.append({
                'bbox': (0, 0, w, h),
                'poly_pts': None,
                'raw_corners': None,
                'master_kpts': None
            })

        # Transform các vùng Ignore Zone dựa trên PRS
        ignore_zones = recipe.get('ignore_zones', [])
        if not ignore_zones and 'all_zones' in recipe:
            ignore_zones = [z for z in recipe.get('all_zones', []) if z.get('is_ignore')]

        target_ignore_polies = []
        for izone in ignore_zones:
            if izone.get("type") == "polygon":
                pts = np.array(izone.get("points", []), dtype=np.float32)
                if len(pts) >= 3:
                    pts_trans = cv2.transform(np.array([pts]), M)[0].astype(np.int32)
                    target_ignore_polies.append(pts_trans)
            else:
                rect = izone.get('rect', [0, 0, 0, 0])
                rx, ry, rw_z, rh_z = rect[0], rect[1], rect[2], rect[3]
                corners = np.float32([[rx, ry], [rx + rw_z, ry], [rx + rw_z, ry + rh_z], [rx, ry + rh_z]])
                corners_trans = cv2.transform(np.array([corners]), M)[0].astype(np.int32)
                target_ignore_polies.append(corners_trans)

        if target_ignore_polies:
            self.log_signal.emit(f"  🚫 Đã định vị {len(target_ignore_polies)} vùng Ignore Zone")

        class_id = self.config.get('class_id', 0)
        label_mode = self.config.get('label_mode', 'seg')
        zone_as_seg = self.config.get('zone_as_seg', False)

        # Nếu chọn chế độ YOLO-Pose: Box chính là vùng Defect Zone, Keypoints tự chấm và đi theo PRS
        if label_mode == 'pose':
            annotations = []
            preview_poses = []
            pose_kpt_mode = self.config.get('pose_kpt_mode', 'two_points')
            recipe_kpts = recipe.get('num_pose_kpts')
            if recipe_kpts is not None:
                num_kpts = int(recipe_kpts)
            else:
                num_kpts = int(self.config.get('num_kpts', 2))
            margin = int(self.config.get('margin', 0))

            for roi in target_rois:
                annot_line, preview_dict = compute_pose_for_roi(
                    roi=roi,
                    img_gray=img_gray,
                    img_w=w,
                    img_h=h,
                    class_id=class_id,
                    pose_kpt_mode=pose_kpt_mode,
                    num_kpts=num_kpts,
                    margin=margin,
                    config=self.config,
                    M=M
                )
                annotations.append(annot_line)
                preview_poses.append(preview_dict)

            self.log_signal.emit(f"  🏷 Đã tạo {len(annotations)} nhãn YOLO-Pose (BBox = Defect Zone, {num_kpts} keypoints theo PRS)")
            success = len(annotations) > 0
            return success, img_bgr, annotations, preview_poses

        # Nếu chọn tùy chọn: Đường vùng lỗi chính là đường segmentation để label
        if zone_as_seg:
            annotations = []
            preview_polies = []
            for roi in target_rois:
                if roi.get('poly_pts') is not None:
                    pts_final = roi['poly_pts']
                elif roi.get('raw_corners') is not None:
                    pts_final = roi['raw_corners']
                else:
                    bx, by, bw, bh = roi['bbox']
                    pts_final = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.int32)

                if len(pts_final) >= 3:
                    if target_ignore_polies:
                        # Cắt bỏ phần Ignore Zone khỏi Defect Zone, lấy đường biên nối tiếp
                        bx_f, by_f, bw_f, bh_f = cv2.boundingRect(pts_final)
                        mx1 = max(0, bx_f - 2)
                        my1 = max(0, by_f - 2)
                        mx2 = min(w, bx_f + bw_f + 2)
                        my2 = min(h, by_f + bh_f + 2)
                        mw = mx2 - mx1
                        mh = my2 - my1
                        if mw > 2 and mh > 2:
                            z_mask = np.zeros((mh, mw), dtype=np.uint8)
                            cv2.fillPoly(z_mask, [pts_final - np.array([mx1, my1])], 255)
                            for ipoly in target_ignore_polies:
                                cv2.fillPoly(z_mask, [ipoly - np.array([mx1, my1])], 0)
                            cnts, _ = cv2.findContours(z_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if cnts:
                                largest_cnt = max(cnts, key=cv2.contourArea)
                                eps = max(1.0, 0.005 * cv2.arcLength(largest_cnt, True))
                                approx = cv2.approxPolyDP(largest_cnt, eps, True)
                                if len(approx) >= 3:
                                    pts_final = (approx.squeeze(1) if len(approx.shape) == 3 else approx) + np.array([mx1, my1])

                    preview_polies.append(pts_final.astype(np.float32).tolist())
                    points_str = []
                    for pt in pts_final:
                        nx = max(0.0, min(1.0, float(pt[0]) / w))
                        ny = max(0.0, min(1.0, float(pt[1]) / h))
                        points_str.append(f"{nx:.6f} {ny:.6f}")
                    annotations.append(f"{class_id} " + " ".join(points_str))

            self.log_signal.emit(f"  🏷 Đã xuất {len(annotations)} polygon trực tiếp từ đường vùng lỗi (Defect Zone)")
            success = len(annotations) > 0
            return success, img_bgr, annotations, preview_polies

        # 3. Dò đường biên từ ngoài vào trong (Outside-In Edge Tracing):
        # - Quét từ đường biên Defect Zone (ngoài) hướng về tâm (trong).
        # - Nếu đụng đường viền của Ignore Zone -> lấy ngay đường viền tại vị trí đụng.
        # - Nếu gặp chuyển mức (mặc định từ Đen sang Trắng) -> lấy điểm chuyển mức.
        # - Đoạn nào không đáp ứng chuyển mức -> tự động lấy đường biên của Defect Zone.
        th_val = self.config.get('threshold', 128)
        use_otsu = self.config.get('use_otsu', False)
        min_area = self.config.get('min_area', 10)
        epsilon_ratio = self.config.get('epsilon_ratio', 0.005)
        margin = self.config.get('margin', 5)
        polarity = self.config.get('polarity', 'black_to_white')
        search_direction = self.config.get('search_direction', 'outside_in')
        dzone_fallback = self.config.get('dzone_fallback', True)
        search_dist = self.config.get('search_dist', 100)

        annotations = []
        preview_polies = []

        for roi in target_rois:
            if roi.get('poly_pts') is not None:
                dzone_poly = roi['poly_pts'].astype(np.float32)
            elif roi.get('raw_corners') is not None:
                dzone_poly = roi['raw_corners'].astype(np.float32)
            else:
                bx, by, bw, bh = roi['bbox']
                dzone_poly = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.float32)

            bx, by, bw, bh = cv2.boundingRect(dzone_poly.astype(np.int32))
            x1 = max(0, bx - margin)
            y1 = max(0, by - margin)
            x2 = min(w, bx + bw + margin)
            y2 = min(h, by + bh + margin)
            cw = x2 - x1
            ch = y2 - y1
            if cw < 4 or ch < 4:
                continue

            patch = img_gray[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(patch, (3, 3), 0)

            thresh_mode = cv2.THRESH_BINARY
            if use_otsu:
                _, bin_patch = cv2.threshold(blurred, 0, 255, thresh_mode | cv2.THRESH_OTSU)
            else:
                _, bin_patch = cv2.threshold(blurred, th_val, 255, thresh_mode)

            # Chuyển đổi tọa độ sang patch
            local_dzone = dzone_poly - np.array([x1, y1], dtype=np.float32)
            local_ignores = [ipoly.astype(np.float32) - np.array([x1, y1], dtype=np.float32) for ipoly in target_ignore_polies]

            pts_local = find_outside_in_segmentation(
                bin_patch,
                local_dzone,
                ignore_polies=local_ignores,
                n_rays=360,
                epsilon_ratio=epsilon_ratio,
                polarity=polarity,
                search_direction=search_direction,
                dzone_fallback=dzone_fallback,
                search_dist=search_dist
            )

            if len(pts_local) >= 3:
                pts_abs = pts_local + np.array([x1, y1], dtype=np.float32)
                area = cv2.contourArea(pts_abs.astype(np.float32))
                if area >= min_area:
                    preview_polies.append(pts_abs.astype(np.float32).tolist())
                    points_str = []
                    for pt in pts_abs:
                        nx = max(0.0, min(1.0, float(pt[0]) / w))
                        ny = max(0.0, min(1.0, float(pt[1]) / h))
                        points_str.append(f"{nx:.6f} {ny:.6f}")
                    annotations.append(f"{class_id} " + " ".join(points_str))
                    self.log_signal.emit(f"  🎯 Đã dò đường biên Outside-In: Diện tích={area:.1f} px ({len(pts_abs)} đỉnh)")

        success = len(annotations) > 0
        return success, img_bgr, annotations, preview_polies

    def run(self):
        files = self.config.get('input_files', [])
        total = len(files)
        output_folder = self.config.get('output_folder', '')
        labels_folder = self.config.get('labels_folder', '')
        if not labels_folder:
            labels_folder = os.path.join(output_folder, 'labels')
        os.makedirs(labels_folder, exist_ok=True)

        save_images = self.config.get('save_images', False)
        images_folder = os.path.join(output_folder, 'images') if save_images else ""
        if save_images:
            os.makedirs(images_folder, exist_ok=True)

        label_mode = self.config.get('label_mode', 'seg')
        is_pose = (label_mode == 'pose')
        if is_pose:
            num_kpts = int(self.config.get('num_kpts', 2))
            kpt_mode = self.config.get('pose_kpt_mode', 'two_points')
            self.log_signal.emit(f"═══ Bắt đầu PRS + Defect Zone YOLO-Pose: {total} ảnh ═══")
            self.log_signal.emit(f"  Thư mục nhãn: {labels_folder}")
            self.log_signal.emit(f"  Class ID: {self.config.get('class_id', 0)}")
            self.log_signal.emit(f"  Chế độ Pose: {kpt_mode} ({num_kpts} điểm keypoints theo PRS)")
            self.log_signal.emit(f"  Bounding Box: Khung Defect Zone (Margin: {self.config.get('margin', 0)} px)")
        else:
            self.log_signal.emit(f"═══ Bắt đầu PRS + Defect Zone YOLO-Seg: {total} ảnh ═══")
            self.log_signal.emit(f"  Thư mục nhãn: {labels_folder}")
            self.log_signal.emit(f"  Class ID: {self.config.get('class_id', 0)}")
            if self.config.get('zone_as_seg', False):
                self.log_signal.emit("  Chế độ: Đường vùng lỗi chính là đường segmentation (Bỏ qua threshold)")
            else:
                self.log_signal.emit(f"  Ngưỡng segment: {'Otsu' if self.config.get('use_otsu', False) else self.config.get('threshold', 128)}")
        self.log_signal.emit("")

        success_count = 0
        for i, filepath in enumerate(files):
            if not self.is_running:
                self.log_signal.emit("⏹ Đã dừng tiến trình.")
                break

            name = os.path.basename(filepath)
            base = os.path.splitext(name)[0]
            self.log_signal.emit(f"[{i + 1}/{total}] 📷 {name}")
            self.progress_signal.emit(i + 1, total)

            success, out_img, annots, polies = self.process_single_image(filepath)
            if success:
                txt_path = os.path.join(labels_folder, f"{base}.txt")
                current_class_id = self.config.get('class_id', 0)

                # Đọc các nhãn cũ nếu file đã có từ các lần chạy trước
                existing_lines = []
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                l_str = line.strip()
                                if not l_str:
                                    continue
                                parts = l_str.split()
                                try:
                                    line_cid = int(parts[0])
                                    # Giữ lại nhãn của các class KHÁC
                                    # Nhãn cũ của class hiện tại sẽ được cập nhật bằng kết quả mới
                                    if line_cid != current_class_id:
                                        existing_lines.append(l_str)
                                except (ValueError, IndexError):
                                    existing_lines.append(l_str)
                    except Exception as e:
                        self.log_signal.emit(f"  ⚠ Lỗi đọc nhãn cũ {base}.txt: {e}")

                combined_annots = existing_lines + annots
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(combined_annots) + '\n')

                if save_images and out_img is not None:
                    out_img_path = os.path.join(images_folder, name)
                    cv2_imwrite(out_img_path, out_img)

                success_count += 1
                unit_name = "nhãn YOLO-Pose" if is_pose else "polygon"
                if existing_lines:
                    self.log_signal.emit(f"  ✅ Đã gộp thêm {len(annots)} {unit_name} (Class {current_class_id}) vào {base}.txt (Tổng: {len(combined_annots)} nhãn)")
                else:
                    self.log_signal.emit(f"  ✅ Đã xuất {len(annots)} {unit_name} vào {base}.txt")
                self.preview_signal.emit(out_img, filepath, polies)
            else:
                self.log_signal.emit(f"  ⚠ Không tìm thấy đối tượng trong {name}")

        # Tự động cập nhật file classes.txt trong labels_folder
        classes_list = self.config.get('classes', [])
        import re
        if classes_list and os.path.isdir(labels_folder):
            classes_path = os.path.join(labels_folder, 'classes.txt')
            try:
                with open(classes_path, 'w', encoding='utf-8') as f:
                    for c_name in classes_list:
                        clean_name = re.sub(r'^\d+\s*[-:]\s*', '', c_name).strip()
                        f.write(f"{clean_name if clean_name else c_name}\n")
                self.log_signal.emit(f"📄 Đã cập nhật classes.txt ({len(classes_list)} classes)")
            except Exception:
                pass

        # Tự động tạo / cập nhật file data.yaml trong output_folder
        if output_folder and os.path.isdir(output_folder):
            yaml_path = os.path.join(output_folder, 'data.yaml')
            try:
                clean_names = []
                for c_name in classes_list:
                    clean_name = re.sub(r'^\d+\s*[-:]\s*', '', c_name).strip()
                    clean_names.append(clean_name if clean_name else c_name)
                if not clean_names:
                    clean_names = ['defect']

                with open(yaml_path, 'w', encoding='utf-8') as yf:
                    yf.write(f"path: {output_folder.replace('\\\\', '/')}\n")
                    yf.write("train: images\n")
                    yf.write("val: images\n\n")
                    if is_pose:
                        num_kpts = int(self.config.get('num_kpts', 2))
                        yf.write(f"kpt_shape: [{num_kpts}, 3]\n")
                        yf.write(f"flip_idx: {list(range(num_kpts))}\n\n")
                    yf.write("names:\n")
                    for idx, c_name in enumerate(clean_names):
                        yf.write(f"  {idx}: {c_name}\n")
                self.log_signal.emit(f"📄 Đã cập nhật data.yaml ({'YOLO-Pose' if is_pose else 'YOLO-Seg'})")
            except Exception:
                pass

        self.log_signal.emit(f"\n═══ Hoàn tất: {success_count}/{total} ảnh thành công ═══")
        self.finished_signal.emit(success_count, total)

    def stop(self):
        self.is_running = False


# ==========================================
# THREAD XỬ LÝ BATCH CROP DEFECT ZONES
# ==========================================
class BatchCropWorkerThread(QThread):
    """
    Luồng chạy nền: Cắt hàng loạt các vùng Defect Zone từ tập ảnh
    1. Đọc từng file ảnh (hỗ trợ cả ảnh BGR và Grayscale).
    2. Căn chỉnh PRS (nếu bật use_prs):
       - Tìm kiếm Pattern 1 và Pattern 2 trong vùng Search tương ứng.
       - Căn chỉnh Affine Transform (xoay + dịch chuyển) để đưa ảnh về hệ tọa độ của Master.
    3. Cắt (crop) từng vùng Defect Zone đã chọn:
       - Hỗ trợ cả hình chữ nhật (Rect) và đa giác (Polygon).
       - Thêm viền đệm (Margin/Padding).
       - Tùy chọn Masking (xóa đen phần ngoài đa giác).
       - Tùy chọn Resize kích thước cố định.
    4. Lưu file ảnh crop vào thư mục output theo quy tắc đặt tên.
    5. Emit signal cập nhật tiến trình và preview.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)               # current, total
    preview_signal = pyqtSignal(object, str, list)       # aligned_img, filepath, list of (zone_name, crop_img, out_path, rect_coords, poly_pts)
    finished_signal = pyqtSignal(int, int, int)          # success_images, total_images, total_crops_saved

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

        recipe = config.get('recipe', {})
        self.tpl1_img = b64_to_img(recipe.get('prs1', {}).get('tpl_b64', ''))
        self.tpl2_img = b64_to_img(recipe.get('prs2', {}).get('tpl_b64', ''))

    def find_best_match(self, img_gray, search_rect, tpl_img):
        """Tìm vị trí khớp tốt nhất của pattern trong search area"""
        if tpl_img is None or tpl_img.size == 0:
            return 0.0, None
        h, w = img_gray.shape[:2]
        if search_rect:
            sx, sy, sw, sh = search_rect
            sx, sy = max(0, min(sx, w - 1)), max(0, min(sy, h - 1))
            sw = max(1, min(sw, w - sx))
            sh = max(1, min(sh, h - sy))
        else:
            sx, sy, sw, sh = 0, 0, w, h

        search_area = img_gray[sy:sy + sh, sx:sx + sw]
        tpl_h, tpl_w = tpl_img.shape[:2]
        if search_area.shape[0] < tpl_h or search_area.shape[1] < tpl_w:
            return 0.0, None

        res = cv2.matchTemplate(search_area, tpl_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        global_pt = (sx + max_loc[0], sy + max_loc[1])
        return float(max_val), global_pt

    def process_single_image(self, filepath):
        """
        Căn chỉnh PRS và crop các Defect Zones từ 1 ảnh.
        Returns: (success, aligned_img, crops_list)
        crops_list: [(zone_name, crop_img, out_path, (x1, y1, cw, ch), poly_pts), ...]
        """
        try:
            # Hỗ trợ đường dẫn tiếng Việt / Unicode trên Windows
            img = None
            try:
                data = np.fromfile(filepath, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                img = cv2_imread(filepath)

            if img is None:
                self.log_signal.emit(f"  ⚠ Không đọc được file ảnh: {os.path.basename(filepath)}")
                return False, None, []

            h, w = img.shape[:2]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            use_prs = self.config.get('use_prs', True)
            prs_fallback = self.config.get('prs_fallback', 'skip')
            aligned_img = img

            if use_prs:
                recipe = self.config.get('recipe', {})
                prs1 = recipe.get('prs1', {})
                prs2 = recipe.get('prs2', {})
                th1 = prs1.get('threshold', 0.6)
                th2 = prs2.get('threshold', 0.6)

                has_prs1 = self.tpl1_img is not None and self.tpl1_img.size > 0
                has_prs2 = self.tpl2_img is not None and self.tpl2_img.size > 0

                val1, pt1 = 0.0, None
                val2, pt2 = 0.0, None
                if has_prs1:
                    val1, pt1 = self.find_best_match(img_gray, prs1.get('search_rect'), self.tpl1_img)
                    self.log_signal.emit(f"  PRS 1 Match: {val1:.3f} (Ngưỡng: {th1:.2f})")
                if has_prs2:
                    val2, pt2 = self.find_best_match(img_gray, prs2.get('search_rect'), self.tpl2_img)
                    self.log_signal.emit(f"  PRS 2 Match: {val2:.3f} (Ngưỡng: {th2:.2f})")

                if has_prs1 and has_prs2 and val1 >= th1 and val2 >= th2:
                    # Affine warp target image to Master orientation
                    src_pts = np.float32([pt1, pt2])
                    dst_pts = np.float32([prs1.get('ref_pt', pt1), prs2.get('ref_pt', pt2)])
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M is not None:
                        aligned_img = cv2.warpAffine(img, M, (w, h))
                        self.log_signal.emit("  ✅ Đã khớp 2 PRS (Affine alignment về Master)")
                    else:
                        self.log_signal.emit("  ⚠ Affine thất bại, chuyển sang tịnh tiến theo PRS 1")
                        ref_pt1 = prs1.get('ref_pt', pt1)
                        dx = float(pt1[0] - ref_pt1[0])
                        dy = float(pt1[1] - ref_pt1[1])
                        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
                        aligned_img = cv2.warpAffine(img, M, (w, h))
                elif has_prs1 and val1 >= th1 and pt1 is not None:
                    ref_pt1 = prs1.get('ref_pt', pt1)
                    dx = float(pt1[0] - ref_pt1[0])
                    dy = float(pt1[1] - ref_pt1[1])
                    M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
                    aligned_img = cv2.warpAffine(img, M, (w, h))
                    self.log_signal.emit(f"  ✅ Đã khớp 1 PRS (Dịch chuyển dx={int(dx)}, dy={int(dy)})")
                else:
                    if prs_fallback == 'skip':
                        self.log_signal.emit("  ❌ PRS không đạt ngưỡng -> Bỏ qua ảnh này")
                        return False, None, []
                    else:
                        self.log_signal.emit("  ⚠ PRS không đạt ngưỡng -> Dùng tọa độ gốc Master")
                        aligned_img = img

            recipe = self.config.get('recipe', {})
            all_defect_zones = recipe.get('defect_zones', [])
            selected_indices = self.config.get('selected_zone_indices', None)
            output_folder = self.config.get('output_folder', '')
            ext = self.config.get('image_format', '.png')
            if not ext.startswith('.'):
                ext = '.' + ext
            naming_mode = self.config.get('naming_mode', 'img_zone')
            margin = self.config.get('margin', 0)
            mask_poly = self.config.get('mask_polygon', False)
            resize_enabled = self.config.get('resize_enabled', False)
            resize_w = self.config.get('resize_w', 224)
            resize_h = self.config.get('resize_h', 224)

            base_name = os.path.splitext(os.path.basename(filepath))[0]
            crops_list = []

            for idx, zone in enumerate(all_defect_zones):
                if selected_indices is not None and idx not in selected_indices:
                    continue

                zone_name = zone.get('name', f"Zone_{idx + 1}").strip()
                # Loại bỏ ký tự không hợp lệ trong tên file
                safe_zone_name = "".join(c for c in zone_name if c.isalnum() or c in (' ', '_', '-')).strip()
                if not safe_zone_name:
                    safe_zone_name = f"Zone_{idx + 1}"

                poly_pts = None
                if zone.get('type') == 'polygon':
                    pts = np.array(zone.get('points', []), dtype=np.int32)
                    if len(pts) >= 3:
                        bx, by, bw, bh = cv2.boundingRect(pts)
                        poly_pts = pts
                    else:
                        continue
                else:
                    r = zone.get('rect', [0, 0, w, h])
                    bx, by, bw, bh = int(r[0]), int(r[1]), int(r[2]), int(r[3])

                x1 = max(0, bx - margin)
                y1 = max(0, by - margin)
                x2 = min(w, bx + bw + margin)
                y2 = min(h, by + bh + margin)
                cw = x2 - x1
                ch = y2 - y1

                if cw <= 0 or ch <= 0:
                    continue

                crop = aligned_img[y1:y2, x1:x2].copy()

                if mask_poly and poly_pts is not None:
                    mask = np.zeros((ch, cw), dtype=np.uint8)
                    local_pts = poly_pts - np.array([x1, y1], dtype=np.int32)
                    cv2.fillPoly(mask, [local_pts], 255)
                    if len(crop.shape) == 3:
                        crop = cv2.bitwise_and(crop, crop, mask=mask)
                    else:
                        crop = cv2.bitwise_and(crop, mask)

                if resize_enabled and resize_w > 0 and resize_h > 0:
                    interp = cv2.INTER_AREA if (crop.shape[1] > resize_w or crop.shape[0] > resize_h) else cv2.INTER_LINEAR
                    crop = cv2.resize(crop, (resize_w, resize_h), interpolation=interp)

                # Xác định đường dẫn file lưu
                if naming_mode == "zone_folder":
                    z_dir = os.path.join(output_folder, safe_zone_name)
                    os.makedirs(z_dir, exist_ok=True)
                    out_path = os.path.join(z_dir, f"{base_name}{ext}")
                elif naming_mode == "img_idx":
                    out_path = os.path.join(output_folder, f"{base_name}_z{idx + 1}{ext}")
                elif naming_mode == "zone_img":
                    out_path = os.path.join(output_folder, f"{safe_zone_name}_{base_name}{ext}")
                else:  # "img_zone"
                    out_path = os.path.join(output_folder, f"{base_name}_{safe_zone_name}{ext}")

                # Ghi file an toàn
                saved = False
                try:
                    ext_code = os.path.splitext(out_path)[1]
                    is_ok, buf = cv2.imencode(ext_code, crop)
                    if is_ok:
                        with open(out_path, 'wb') as f:
                            f.write(buf)
                        saved = True
                except Exception:
                    pass
                if not saved:
                    cv2_imwrite(out_path, crop)

                crops_list.append((safe_zone_name, crop, out_path, (x1, y1, cw, ch), poly_pts))

            return True, aligned_img, crops_list

        except Exception as e:
            self.log_signal.emit(f"  ❌ Lỗi xử lý {os.path.basename(filepath)}: {e}")
            return False, None, []

    def run(self):
        files = self.config.get('input_files', [])
        total = len(files)
        output_folder = self.config.get('output_folder', '')
        os.makedirs(output_folder, exist_ok=True)

        self.log_signal.emit(f"═══ Bắt đầu Batch Crop Defect Zones: {total} ảnh ═══")
        self.log_signal.emit(f"  Thư mục lưu: {output_folder}")
        self.log_signal.emit(f"  Format: {self.config.get('image_format', '.png')}")
        self.log_signal.emit(f"  Margin: {self.config.get('margin', 0)} px")
        self.log_signal.emit(f"  Căn chỉnh PRS: {'BẬT' if self.config.get('use_prs', True) else 'TẮT'}")
        self.log_signal.emit("")

        success_img_count = 0
        total_crops_saved = 0

        for i, filepath in enumerate(files):
            if not self.is_running:
                self.log_signal.emit("⏹ Đã dừng tiến trình crop.")
                break

            name = os.path.basename(filepath)
            self.log_signal.emit(f"[{i + 1}/{total}] 📷 {name}")
            self.progress_signal.emit(i + 1, total)

            success, aligned_img, crops = self.process_single_image(filepath)
            if success and crops:
                success_img_count += 1
                total_crops_saved += len(crops)
                self.log_signal.emit(f"  ✅ Đã crop thành công {len(crops)} vùng defect zone")
                self.preview_signal.emit(aligned_img, filepath, crops)
            elif success and not crops:
                self.log_signal.emit(f"  ⚠ Không có vùng defect zone nào được chọn để crop")
            else:
                self.log_signal.emit(f"  ⚠ Không crop được ảnh: {name}")

        self.log_signal.emit(f"\n═══ Hoàn tất Batch Crop: {success_img_count}/{total} ảnh thành công, tổng cộng {total_crops_saved} ảnh crop ═══")
        self.finished_signal.emit(success_img_count, total, total_crops_saved)

    def stop(self):
        self.is_running = False


# ==========================================
# HELPER: TÌM FILE LABEL & ĐỌC CLASS NAMES
# ==========================================
def find_yolo_label_file(img_path):
    """
    Tìm file nhãn .txt tương ứng cho 1 file ảnh, hỗ trợ:
    1. Cùng thư mục với ảnh (flat)
    2. Trong thư mục con 'labels/'
    3. Trong cấu trúc YOLO chuẩn 'images/' -> 'labels/'
    4. Thư mục sibling 'labels/'
    """
    d = os.path.dirname(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]

    # 1. Cùng thư mục
    c1 = os.path.join(d, f"{base}.txt")
    if os.path.isfile(c1):
        return c1

    # 2. Thư mục con labels/
    c2 = os.path.join(d, "labels", f"{base}.txt")
    if os.path.isfile(c2):
        return c2

    # 3. Thay thế 'images' bằng 'labels' trong đường dẫn
    norm_path = os.path.normpath(img_path)
    parts = norm_path.split(os.sep)
    if 'images' in parts:
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts_labels = list(parts)
        parts_labels[idx] = 'labels'
        parts_labels[-1] = f"{base}.txt"
        c3 = os.sep.join(parts_labels)
        if os.path.isfile(c3):
            return c3

    # 4. Thư mục sibling labels/ của thư mục cha
    parent_d = os.path.dirname(d)
    c4 = os.path.join(parent_d, "labels", f"{base}.txt")
    if os.path.isfile(c4):
        return c4

    return None


def read_yolo_class_names_from_dataset(dataset_dir):
    """Đọc tên class từ classes.txt hoặc data.yaml trong thư mục dataset nếu có"""
    class_names = {}

    # 1. Thử classes.txt
    classes_txt_candidates = [
        os.path.join(dataset_dir, "classes.txt"),
        os.path.join(dataset_dir, "labels", "classes.txt"),
    ]
    for c_txt in classes_txt_candidates:
        if os.path.isfile(c_txt):
            try:
                with open(c_txt, 'r', encoding='utf-8', errors='ignore') as f:
                    for idx, line in enumerate(f):
                        name = line.strip()
                        if name:
                            class_names[idx] = name
                if class_names:
                    return class_names
            except Exception:
                pass

    # 2. Thử data.yaml hoặc dataset.yaml
    yaml_candidates = [
        os.path.join(dataset_dir, "data.yaml"),
        os.path.join(dataset_dir, "dataset.yaml"),
    ]
    for y_path in yaml_candidates:
        if os.path.isfile(y_path):
            try:
                import re
                with open(y_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                in_names = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('names:'):
                        in_names = True
                        rest = stripped[6:].strip()
                        if rest.startswith('[') and rest.endswith(']'):
                            items = [x.strip().strip("'\"") for x in rest[1:-1].split(',')]
                            for i, it in enumerate(items):
                                if it:
                                    class_names[i] = it
                            return class_names
                        continue
                    if in_names:
                        if stripped.startswith('#') or not stripped:
                            continue
                        m_dict = re.match(r"^(\d+)\s*:\s*(.+)$", stripped)
                        if m_dict:
                            class_names[int(m_dict.group(1))] = m_dict.group(2).strip().strip("'\"")
                        elif stripped.startswith('-'):
                            val = stripped[1:].strip().strip("'\"")
                            class_names[len(class_names)] = val
                        elif not line.startswith(' ') and not line.startswith('\t'):
                            in_names = False
                if class_names:
                    return class_names
            except Exception:
                pass

    return class_names


# ==========================================
# THREAD QUÉT DATASET (DATASET EXPLORER)
# ==========================================
class DatasetScanWorkerThread(QThread):
    """Luồng quét và thống kê chi tiết các dataset YOLO gốc"""
    progress_signal = pyqtSignal(int, int)       # current, total
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)           # full scanned info

    def __init__(self, dataset_paths):
        super().__init__()
        self.dataset_paths = dataset_paths
        self.is_running = True

    def run(self):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        datasets_info = []
        total_images_all = 0
        total_objects_all = 0
        all_class_counts = {}
        all_class_images_counts = {}
        all_class_names = {}
        empty_images_count = 0

        self.log_signal.emit(f"🔍 Bắt đầu quét {len(self.dataset_paths)} thư mục dataset...")

        for ds_idx, dpath in enumerate(self.dataset_paths):
            if not self.is_running:
                break
            if not os.path.isdir(dpath):
                self.log_signal.emit(f"  ⚠ Bỏ qua: Thư mục không tồn tại: {dpath}")
                continue

            ds_name = os.path.basename(os.path.normpath(dpath))
            self.log_signal.emit(f"  📁 Quét dataset: {ds_name}...")

            detected_names = read_yolo_class_names_from_dataset(dpath)
            for cid, cname in detected_names.items():
                all_class_names[cid] = cname

            img_files = []
            for root, _, files in os.walk(dpath):
                if os.path.basename(root).lower() == 'labels':
                    continue
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        img_files.append(os.path.join(root, f))

            ds_img_records = []
            ds_class_counts = {}
            total_ds_imgs = len(img_files)

            for idx, img_p in enumerate(img_files):
                if not self.is_running:
                    break
                if idx % 50 == 0:
                    self.progress_signal.emit(idx + 1, total_ds_imgs)

                lbl_p = find_yolo_label_file(img_p)
                img_classes = set()
                img_counts = {}

                if lbl_p and os.path.isfile(lbl_p):
                    try:
                        with open(lbl_p, 'r', encoding='utf-8', errors='ignore') as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if parts:
                                    try:
                                        cid = int(parts[0])
                                        img_classes.add(cid)
                                        img_counts[cid] = img_counts.get(cid, 0) + 1
                                        ds_class_counts[cid] = ds_class_counts.get(cid, 0) + 1
                                        all_class_counts[cid] = all_class_counts.get(cid, 0) + 1
                                        total_objects_all += 1
                                        if cid not in all_class_names:
                                            all_class_names[cid] = f"class_{cid}"
                                    except ValueError:
                                        pass
                    except Exception:
                        pass

                if not img_classes:
                    empty_images_count += 1

                for cid in img_classes:
                    all_class_images_counts[cid] = all_class_images_counts.get(cid, 0) + 1

                ds_img_records.append({
                    'img_path': img_p,
                    'label_path': lbl_p,
                    'classes': list(img_classes),
                    'counts': img_counts,
                    'filename': os.path.basename(img_p),
                    'dataset_name': ds_name,
                    'dataset_path': dpath
                })

            total_images_all += len(ds_img_records)
            datasets_info.append({
                'path': dpath,
                'name': ds_name,
                'img_count': len(ds_img_records),
                'class_counts': ds_class_counts,
                'images': ds_img_records,
                'class_names': detected_names
            })

            self.log_signal.emit(f"    ✔ {ds_name}: {len(ds_img_records)} ảnh, {sum(ds_class_counts.values())} vật thể")

        result = {
            'datasets': datasets_info,
            'total_images': total_images_all,
            'total_objects': total_objects_all,
            'all_class_counts': all_class_counts,
            'all_class_images_counts': all_class_images_counts,
            'all_class_names': all_class_names,
            'empty_images_count': empty_images_count
        }

        self.log_signal.emit(f"✅ Hoàn tất quét! Tổng cộng: {total_images_all} ảnh, {total_objects_all} vật thể ({len(all_class_counts)} classes).")
        self.finished_signal.emit(result)

    def stop(self):
        self.is_running = False


# ==========================================
# THREAD TẠO DATASET MỚI (DATASET BUILDER)
# ==========================================
class DatasetBuilderWorkerThread(QThread):
    """
    Luồng chạy nền: Tạo dataset mới từ các dataset nguồn theo cấu hình phân bổ
    1. Chọn mẫu ảnh thỏa mãn phân bổ riêng theo % từng nguồn hoặc target_counts.
    2. Chia thành các tập train / val / test theo tỉ lệ.
    3. Copy ảnh và file nhãn .txt vào cấu trúc YOLO chuẩn.
    4. Tự động sinh file data.yaml.
    """
    progress_signal = pyqtSignal(int, int)        # current, total
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, dict) # success, message, summary_dict

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def run(self):
        import shutil
        import random
        try:
            available_images = self.config.get('available_images', [])
            raw_source_percentages = self.config.get('source_percentages', {})
            source_percentages = {os.path.normpath(k): v for k, v in raw_source_percentages.items()}
            mode = self.config.get('mode', 'percentage')
            target_counts = self.config.get('target_counts', {})
            split_ratios = self.config.get('split_ratios', {'train': 80, 'val': 10, 'test': 10})
            output_dir = self.config.get('output_dir', '')
            class_names = self.config.get('class_names', {})
            include_bg = self.config.get('include_bg', False)
            bg_target_count = self.config.get('bg_count', 0)

            os.makedirs(output_dir, exist_ok=True)
            self.log_signal.emit(f"═══ Bắt đầu tạo Dataset Mới ═══")
            self.log_signal.emit(f"  Thư mục đích: {output_dir}")
            self.log_signal.emit(f"  Chế độ: {mode}")
            self.log_signal.emit(f"  Tỉ lệ Split: Train {split_ratios.get('train', 80)}% | Val {split_ratios.get('val', 10)}% | Test {split_ratios.get('test', 10)}%")

            shuffled_pool = list(available_images)
            random.shuffle(shuffled_pool)

            if mode == 'all':
                selected_records = list(shuffled_pool)
            elif mode == 'percentage':
                images_by_source = {}
                for rec in available_images:
                    dpath = rec.get('dataset_path') or rec.get('dataset_name', 'default')
                    dpath = os.path.normpath(dpath) if dpath else dpath
                    if dpath not in images_by_source:
                        images_by_source[dpath] = []
                    images_by_source[dpath].append(rec)

                selected_records = []
                selected_set = set()

                for dpath, img_list in images_by_source.items():
                    pct = source_percentages.get(dpath)
                    if pct is None:
                        pct = source_percentages.get(img_list[0].get('dataset_name'), 100)
                    ds_name = img_list[0].get('dataset_name', os.path.basename(str(dpath)))
                    if pct <= 0:
                        self.log_signal.emit(f"  ⏭ Bỏ qua nguồn '{ds_name}' (tỉ lệ 0%)")
                        continue
                    n_total = len(img_list)
                    k = int(round(n_total * (pct / 100.0)))
                    if pct > 0 and k == 0 and n_total > 0:
                        k = 1
                    k = min(k, n_total)
                    sampled = random.sample(img_list, k)
                    selected_records.extend(sampled)
                    for rec in sampled:
                        selected_set.add(rec['img_path'])
                    self.log_signal.emit(f"  📥 Nguồn '{ds_name}': lấy {k}/{n_total} ảnh ({pct}%)")

                if include_bg and bg_target_count > 0:
                    bg_pool = [r for r in available_images if r['img_path'] not in selected_set and not r.get('classes')]
                    random.shuffle(bg_pool)
                    bg_to_add = bg_pool[:bg_target_count]
                    selected_records.extend(bg_to_add)
                    if bg_to_add:
                        self.log_signal.emit(f"  🖼 Đã bổ sung {len(bg_to_add)} ảnh background (không nhãn)")
            elif mode == 'target_counts':
                selected_records = self._sample_by_targets(shuffled_pool, target_counts, include_bg, bg_target_count)
            else:
                selected_records = list(shuffled_pool)

            if not selected_records:
                self.finished_signal.emit(False, "Không có ảnh nào được chọn thỏa mãn điều kiện!", {})
                return

            self.log_signal.emit(f"  📊 Tổng số ảnh đã chọn lọc: {len(selected_records)} ảnh")
            random.shuffle(selected_records)

            total_selected = len(selected_records)
            r_train = split_ratios.get('train', 80)
            r_val = split_ratios.get('val', 10)
            r_test = split_ratios.get('test', 10)
            sum_r = r_train + r_val + r_test
            if sum_r == 0:
                sum_r = 100

            n_train = int(round(total_selected * (r_train / sum_r)))
            n_val = int(round(total_selected * (r_val / sum_r)))
            n_test = total_selected - n_train - n_val
            if n_test < 0:
                n_test = 0
                n_val = total_selected - n_train

            splits = {
                'train': selected_records[:n_train],
                'val': selected_records[n_train:n_train + n_val],
                'test': selected_records[n_train + n_val:]
            }

            self.log_signal.emit(f"  🍰 Phân bổ: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

            copied_count = 0
            total_to_copy = total_selected
            split_stats = {'train': {}, 'val': {}, 'test': {}}

            # Luôn tạo trước các thư mục train và val (kể cả khi tập rỗng) để đảm bảo tương thích YOLO
            for req_split in ['train', 'val']:
                os.makedirs(os.path.join(output_dir, req_split, "images"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, req_split, "labels"), exist_ok=True)

            for split_name, records in splits.items():
                if not records:
                    continue
                img_dest_dir = os.path.join(output_dir, split_name, "images")
                lbl_dest_dir = os.path.join(output_dir, split_name, "labels")
                os.makedirs(img_dest_dir, exist_ok=True)
                os.makedirs(lbl_dest_dir, exist_ok=True)

                for rec in records:
                    if not self.is_running:
                        self.log_signal.emit("⏹ Đã dừng tiến trình tạo dataset.")
                        self.finished_signal.emit(False, "Tiến trình bị dừng bởi người dùng", {})
                        return

                    src_img = rec['img_path']
                    src_lbl = rec.get('label_path')
                    fname = os.path.basename(src_img)
                    base = os.path.splitext(fname)[0]

                    dest_img = os.path.join(img_dest_dir, fname)
                    dest_lbl = os.path.join(lbl_dest_dir, f"{base}.txt")

                    # Tránh ghi đè nếu trùng tên từ 2 dataset khác nhau
                    counter = 1
                    while os.path.exists(dest_img) or os.path.exists(dest_lbl):
                        fname = f"{base}_{counter}{os.path.splitext(src_img)[1]}"
                        dest_img = os.path.join(img_dest_dir, fname)
                        dest_lbl = os.path.join(lbl_dest_dir, f"{base}_{counter}.txt")
                        counter += 1

                    shutil.copy2(src_img, dest_img)

                    if src_lbl and os.path.isfile(src_lbl):
                        shutil.copy2(src_lbl, dest_lbl)
                    else:
                        with open(dest_lbl, 'w', encoding='utf-8') as f:
                            pass

                    for cid, cnt in rec.get('counts', {}).items():
                        split_stats[split_name][cid] = split_stats[split_name].get(cid, 0) + cnt

                    copied_count += 1
                    self.progress_signal.emit(copied_count, total_to_copy)

            # Tạo file data.yaml
            norm_out = os.path.normpath(output_dir).replace('\\', '/')
            has_test = len(splits['test']) > 0

            used_cids = set()
            for s_dict in split_stats.values():
                used_cids.update(s_dict.keys())
            if not used_cids:
                used_cids = set(class_names.keys())

            used_cids_int = set()
            for c in used_cids:
                try:
                    used_cids_int.add(int(c))
                except (ValueError, TypeError):
                    pass

            max_cid = max(used_cids_int) if used_cids_int else 0
            final_class_dict = {}
            for i in range(max_cid + 1):
                final_class_dict[i] = class_names.get(i, class_names.get(str(i), f"class_{i}"))

            if self.config.get('create_yaml', True):
                yaml_lines = [
                    f"# YOLO Dataset - Tự động tạo bởi Auto Create Defect App",
                    f"path: {norm_out}",
                    f"train: train/images",
                    f"val: val/images",
                ]
                if has_test:
                    yaml_lines.append("test: test/images")
                yaml_lines.append("")
                yaml_lines.append(f"nc: {len(final_class_dict)}")
                yaml_lines.append("names:")
                for cid, cname in sorted(final_class_dict.items()):
                    yaml_lines.append(f"  {cid}: {cname}")

                yaml_path = os.path.join(output_dir, "data.yaml")
                with open(yaml_path, 'w', encoding='utf-8') as yf:
                    yf.write("\n".join(yaml_lines) + "\n")

                self.log_signal.emit(f"  📄 Đã tạo file cấu hình: {yaml_path}")
            self.log_signal.emit(f"═══ HOÀN TẤT: Đã tạo Dataset thành công! ═══")

            summary = {
                'output_dir': output_dir,
                'total_images': total_selected,
                'train_count': len(splits['train']),
                'val_count': len(splits['val']),
                'test_count': len(splits['test']),
                'split_stats': split_stats,
                'class_names': final_class_dict
            }
            self.finished_signal.emit(True, "Tạo dataset thành công!", summary)

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.log_signal.emit(f"❌ Lỗi: {e}")
            self.finished_signal.emit(False, str(e), {})

    def _sample_by_targets(self, pool, target_counts, include_bg, bg_target_count):
        rem_targets = {cid: max(0, int(cnt)) for cid, cnt in target_counts.items() if cnt > 0}
        selected = []
        selected_set = set()

        for rec in pool:
            if not rem_targets:
                break
            img_counts = rec.get('counts', {})
            needs_this = any(cid in rem_targets and rem_targets[cid] > 0 for cid in img_counts)
            if needs_this:
                selected.append(rec)
                selected_set.add(rec['img_path'])
                for cid, cnt in img_counts.items():
                    if cid in rem_targets:
                        rem_targets[cid] -= cnt
                        if rem_targets[cid] <= 0:
                            del rem_targets[cid]

        if include_bg and bg_target_count > 0:
            bg_added = 0
            for rec in pool:
                if bg_added >= bg_target_count:
                    break
                if rec['img_path'] not in selected_set and not rec.get('classes'):
                    selected.append(rec)
                    selected_set.add(rec['img_path'])
                    bg_added += 1

        return selected

    def stop(self):
        self.is_running = False


