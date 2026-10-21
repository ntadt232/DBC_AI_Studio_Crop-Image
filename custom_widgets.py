"""
Custom Widgets cho Defect & Dataset Studio AI.
Mở rộng từ kiến trúc Vision AI Studio.

Thay đổi chính:
- Thêm style DEFECT_ZONE (cam-đỏ dashed) cho ResizableRectItem
- ImageViewer hỗ trợ tool DEFECT_ZONE
- Bỏ RoiPreviewWidget, FailAlertDialog (không cần)
- Thêm DefectPreviewWidget để xem preview defect đã tạo
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsRectItem,
                              QGraphicsItem, QFrame, QVBoxLayout, QLabel)
from PyQt6.QtCore import pyqtSignal, Qt, QRectF, QPointF
from PyQt6.QtGui import QPen, QColor, QBrush, QFont, QPolygonF, QPainter
from utils import numpy_to_qpixmap, cv2_imread


# ==========================================
# CUSTOM ITEM: DEFECT ZONE / PRS RECT CÓ THỂ DI CHUYỂN & RESIZE
# ==========================================
class ResizableRectItem(QGraphicsRectItem):
    """Khung chữ nhật cho phép kéo thả và thay đổi kích thước trên giao diện"""
    handle_size = 10

    def __init__(self, rect, parent_viewer, item_type="DEFECT_ZONE", class_id=None):
        super().__init__(rect)
        self.viewer = parent_viewer
        self.item_type = item_type
        self.class_id = class_id
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # Style theo loại Tool
        if "SEARCH" in item_type:
            # Màu Vàng rực rỡ nét đứt cho vùng Search
            self.setPen(QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine))
            self.setBrush(QBrush(QColor(255, 215, 0, 30)))
        elif "PATTERN" in item_type:
            # Màu Xanh Lam (Cyan/Deep Sky Blue) nét liền đậm cho Pattern
            self.setPen(QPen(QColor(0, 191, 255), 2, Qt.PenStyle.SolidLine))
            self.setBrush(QBrush(QColor(0, 191, 255, 55)))
        elif item_type == "DEFECT_ZONE":
            # Màu cam-đỏ dashed cho defect zone
            self.setPen(QPen(QColor(255, 140, 0), 2, Qt.PenStyle.DashDotLine))
            self.setBrush(QBrush(QColor(255, 100, 0, 35)))
        elif "IGNORE" in item_type:
            # Màu Đỏ nhạt / Đỏ cam viền đứt nét rõ ràng cho Ignore Zone
            self.setPen(QPen(QColor(245, 60, 60), 2, Qt.PenStyle.DashDotLine))
            self.setBrush(QBrush(QColor(245, 60, 60, 45)))
        elif "YOLO_LABEL" in item_type:
            # Màu tùy theo class_id
            colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), 
                      QColor(255, 255, 0), QColor(255, 0, 255)]
            color = colors[int(self.class_id) % len(colors)] if self.class_id is not None else QColor(255, 255, 255)
            self.setPen(QPen(color, 1))
            c = QColor(color)
            c.setAlpha(40)
            self.setBrush(QBrush(c))
        else:
            self.setPen(QPen(QColor(0, 255, 0), 2))
            self.setBrush(QBrush(QColor(0, 255, 0, 30)))

        self.keypoints = []
        self.active_handle = None

    def boundingRect(self):
        r = self.rect()
        pad = 12.0
        label_h = 24.0
        return QRectF(r.left() - pad, r.top() - pad - label_h, r.width() + 2 * pad, r.height() + 2 * pad + label_h)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())
        
        # Nhãn tên cho Search & Pattern & Defect Zone & Ignore Zone để phân biệt rõ ràng
        if "SEARCH" in self.item_type or "PATTERN" in self.item_type or "DEFECT_ZONE" in self.item_type or "IGNORE" in self.item_type:
            label_text = self.item_type.replace("_", " ")
            painter.setPen(QPen(self.pen().color()))
            font = QFont("Arial", 9, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            rect_text = fm.boundingRect(label_text)
            rect_text.adjust(-4, -2, 4, 2)
            top_y = self.rect().y() - rect_text.height()
            if top_y < 0:
                top_y = self.rect().y()
            rect_text.moveTo(int(self.rect().x()), int(top_y))
            painter.fillRect(rect_text, self.pen().color())
            painter.setPen(QPen(QColor(20, 20, 20)))
            painter.drawText(rect_text, Qt.AlignmentFlag.AlignCenter, label_text)

        if self.item_type in ("YOLO_LABEL", "YOLO_LABEL_POSE") and self.class_id is not None:
            names = getattr(self.viewer, 'yolo_class_names', {})
            name = names.get(int(self.class_id), f"Class {self.class_id}")
            painter.setPen(QPen(self.pen().color()))
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            rect_text = fm.boundingRect(name)
            rect_text.adjust(-4, -4, 4, 4)
            rect_text.moveTo(int(self.rect().x()), int(self.rect().y() - rect_text.height()))
            if rect_text.top() < 0:
                rect_text.moveTo(int(self.rect().x()), int(self.rect().y()))
            painter.fillRect(rect_text, self.pen().color())
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect_text, Qt.AlignmentFlag.AlignCenter, name)

        if getattr(self, 'keypoints', None):
            # Vẽ đường nối giữa các keypoints liên tiếp (skeleton line)
            if len(self.keypoints) > 1:
                painter.setPen(QPen(QColor(0, 220, 255, 180), 1.5, Qt.PenStyle.DashLine))
                for i in range(len(self.keypoints) - 1):
                    k1 = self.keypoints[i]
                    k2 = self.keypoints[i + 1]
                    if k1.get('v', 2) > 0 and k2.get('v', 2) > 0:
                        painter.drawLine(k1['pos'], k2['pos'])

            for idx, kpt in enumerate(self.keypoints):
                v = kpt.get('v', 2)
                if v == 0:
                    continue
                pt = kpt['pos']
                dot_color = QColor(0, 255, 128) if v == 2 else (QColor(255, 140, 0) if v == 1 else QColor(220, 50, 50))
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.setBrush(QBrush(dot_color))
                r_pt = QRectF(pt.x() - 4.5, pt.y() - 4.5, 9, 9)
                painter.drawEllipse(r_pt)
                painter.setPen(QPen(QColor(255, 255, 255)))
                font = QFont("Arial", 8, QFont.Weight.Bold)
                painter.setFont(font)
                painter.drawText(QRectF(pt.x() + 6, pt.y() - 10, 24, 20), Qt.AlignmentFlag.AlignLeft, str(idx + 1))

        if self.isSelected():
            r = self.rect()
            pts = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
            hs = 8.0
            painter.setPen(QPen(QColor(255, 255, 255), 1.8))
            painter.setBrush(QBrush(QColor(245, 50, 50)))
            for pt in pts:
                painter.drawRect(QRectF(pt.x() - hs / 2, pt.y() - hs / 2, hs, hs))

    def get_handle_at_pos(self, pos):
        r = self.rect()
        pts = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
        tol = 10.0
        tol = min(tol, max(3.0, min(r.width(), r.height()) * 0.35))
        for i, pt in enumerate(pts):
            if QRectF(pt.x() - tol / 2, pt.y() - tol / 2, tol, tol).contains(pos):
                return i
        return None

    def mousePressEvent(self, event):
        handle = self.get_handle_at_pos(event.pos())
        if handle is not None:
            self.setSelected(True)
            self.active_handle = handle
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event):
        if self.isSelected() and self.get_handle_at_pos(event.pos()) is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor if self.isSelected() else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_handle is not None:
            r = self.rect()
            pos = event.pos()
            if self.active_handle == 0:
                self.setRect(QRectF(pos, r.bottomRight()).normalized())
            elif self.active_handle == 1:
                self.setRect(QRectF(QPointF(r.left(), pos.y()), QPointF(pos.x(), r.bottom())).normalized())
            elif self.active_handle == 2:
                self.setRect(QRectF(r.topLeft(), pos).normalized())
            elif self.active_handle == 3:
                self.setRect(QRectF(QPointF(pos.x(), r.top()), QPointF(r.right(), pos.y())).normalized())

            if self.item_type in ("DEFECT_ZONE", "IGNORE_ZONE"):
                self.viewer.on_zone_geometry_changed(self)
            elif "SEARCH" in self.item_type or "PATTERN" in self.item_type:
                self.viewer.on_prs_geometry_changed(self)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.active_handle is not None:
            self.active_handle = None
            self.ungrabMouse()
        super().mouseReleaseEvent(event)
        if self.item_type in ("DEFECT_ZONE", "IGNORE_ZONE"):
            self.viewer.on_zone_geometry_changed(self)
        elif "SEARCH" in self.item_type or "PATTERN" in self.item_type:
            self.viewer.on_prs_geometry_changed(self)
        event.accept()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.item_type in ("DEFECT_ZONE", "IGNORE_ZONE"):
                self.viewer.on_zone_geometry_changed(self)
            elif "SEARCH" in self.item_type or "PATTERN" in self.item_type:
                self.viewer.on_prs_geometry_changed(self)
        return super().itemChange(change, value)


# ==========================================
# CUSTOM ITEM: DEFECT ZONE POLYGON CÓ THỂ DI CHUYỂN & KÉO ĐỈNH
# ==========================================
from PyQt6.QtWidgets import QGraphicsPolygonItem
from PyQt6.QtGui import QPolygonF

class ResizablePolygonItem(QGraphicsPolygonItem):
    """Khung đa giác cho phép kéo thả và thay đổi đỉnh"""
    handle_size = 11

    def __init__(self, polygon, parent_viewer, item_type="DEFECT_ZONE_POLY", class_id=None):
        super().__init__(polygon)
        self.viewer = parent_viewer
        self.item_type = item_type
        self.class_id = class_id
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        if "YOLO_LABEL" in item_type:
            colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), 
                      QColor(255, 255, 0), QColor(255, 0, 255)]
            color = colors[int(self.class_id) % len(colors)] if self.class_id is not None else QColor(255, 255, 255)
            self.setPen(QPen(color, 1))
            c = QColor(color)
            c.setAlpha(40)
            self.setBrush(QBrush(c))
        elif "IGNORE" in item_type:
            # Màu Đỏ nhạt / Đỏ cam viền đứt nét rõ ràng cho Ignore Poly
            self.setPen(QPen(QColor(245, 60, 60), 2, Qt.PenStyle.DashDotLine))
            self.setBrush(QBrush(QColor(245, 60, 60, 45)))
        else:
            # Màu cam-đỏ dashed cho defect zone
            self.setPen(QPen(QColor(255, 140, 0), 2, Qt.PenStyle.DashDotLine))
            self.setBrush(QBrush(QColor(255, 100, 0, 35)))
            
        self.keypoints = []
        self.active_handle = None

    def boundingRect(self):
        r = self.polygon().boundingRect()
        pad = 12.0
        label_h = 24.0
        return QRectF(r.left() - pad, r.top() - pad - label_h, r.width() + 2 * pad, r.height() + 2 * pad + label_h)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawPolygon(self.polygon())
        
        # Nhãn tên cho Defect Poly và Ignore Poly
        if "IGNORE" in self.item_type or self.item_type == "DEFECT_ZONE_POLY":
            label_text = self.item_type.replace("_", " ")
            painter.setPen(QPen(self.pen().color()))
            font = QFont("Arial", 9, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            rect_text = fm.boundingRect(label_text)
            rect_text.adjust(-4, -2, 4, 2)
            top_y = self.polygon().boundingRect().topLeft().y() - rect_text.height()
            if top_y < 0:
                top_y = self.polygon().boundingRect().topLeft().y()
            rect_text.moveTo(int(self.polygon().boundingRect().topLeft().x()), int(top_y))
            painter.fillRect(rect_text, self.pen().color())
            painter.setPen(QPen(QColor(20, 20, 20)))
            painter.drawText(rect_text, Qt.AlignmentFlag.AlignCenter, label_text)

        if "YOLO_LABEL" in self.item_type and self.class_id is not None:
            names = getattr(self.viewer, 'yolo_class_names', {})
            name = names.get(int(self.class_id), f"Class {self.class_id}")
            painter.setPen(QPen(self.pen().color()))
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            rect = fm.boundingRect(name)
            rect.adjust(-4, -4, 4, 4)
            bg_rect = QRectF(self.polygon().boundingRect().topLeft().x(), self.polygon().boundingRect().topLeft().y() - rect.height(), rect.width(), rect.height())
            painter.fillRect(bg_rect, self.pen().color())
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, name)

        if getattr(self, 'keypoints', None):
            if len(self.keypoints) > 1:
                painter.setPen(QPen(QColor(0, 220, 255, 180), 1.5, Qt.PenStyle.DashLine))
                for i in range(len(self.keypoints) - 1):
                    k1 = self.keypoints[i]
                    k2 = self.keypoints[i + 1]
                    if k1.get('v', 2) > 0 and k2.get('v', 2) > 0:
                        painter.drawLine(k1['pos'], k2['pos'])

            for idx, kpt in enumerate(self.keypoints):
                v = kpt.get('v', 2)
                if v == 0:
                    continue
                pt = kpt['pos']
                dot_color = QColor(0, 255, 128) if v == 2 else (QColor(255, 140, 0) if v == 1 else QColor(220, 50, 50))
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.setBrush(QBrush(dot_color))
                r_pt = QRectF(pt.x() - 4.5, pt.y() - 4.5, 9, 9)
                painter.drawEllipse(r_pt)
                painter.setPen(QPen(QColor(255, 255, 255)))
                font = QFont("Arial", 8, QFont.Weight.Bold)
                painter.setFont(font)
                painter.drawText(QRectF(pt.x() + 6, pt.y() - 10, 24, 20), Qt.AlignmentFlag.AlignLeft, str(idx + 1))

        if self.isSelected():
            poly = self.polygon()
            hs = 8.0
            # Điểm nút to tròn rõ ràng: viền trắng 1.8px, ruột đỏ nổi bật
            painter.setPen(QPen(QColor(255, 255, 255), 1.8))
            painter.setBrush(QBrush(QColor(245, 50, 50)))
            for i in range(poly.count()):
                pt = poly.at(i)
                painter.drawEllipse(QRectF(pt.x() - hs / 2, pt.y() - hs / 2, hs, hs))

    def get_handle_at_pos(self, pos):
        poly = self.polygon()
        tol = 10.0
        for i in range(poly.count()):
            pt = poly.at(i)
            if QRectF(pt.x() - tol / 2, pt.y() - tol / 2, tol, tol).contains(pos):
                return i
        return None

    def mousePressEvent(self, event):
        handle = self.get_handle_at_pos(event.pos())
        if handle is not None:
            self.setSelected(True)
            self.active_handle = handle
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event):
        if self.isSelected() and self.get_handle_at_pos(event.pos()) is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor if self.isSelected() else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_handle is not None:
            poly = self.polygon()
            poly.replace(self.active_handle, event.pos())
            self.setPolygon(poly)
            if self.item_type in ("DEFECT_ZONE_POLY", "IGNORE_ZONE_POLY"):
                self.viewer.on_zone_geometry_changed(self)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.active_handle is not None:
            self.active_handle = None
            self.ungrabMouse()
        super().mouseReleaseEvent(event)
        if self.item_type in ("DEFECT_ZONE_POLY", "IGNORE_ZONE_POLY"):
            self.viewer.on_zone_geometry_changed(self)
        event.accept()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.item_type in ("DEFECT_ZONE_POLY", "IGNORE_ZONE_POLY"):
                self.viewer.on_zone_geometry_changed(self)
        return super().itemChange(change, value)


# ==========================================
# CUSTOM WIDGET: IMAGE VIEWER VỚI ZOOM, PAN, VẼ DEFECT ZONE/PRS
# ==========================================
class ImageViewer(QGraphicsView):
    """Bộ xem ảnh nâng cao hỗ trợ zoom, pan, vẽ Defect Zone/PRS trên ảnh"""
    zone_added_signal = pyqtSignal(QRectF)
    zone_selected_signal = pyqtSignal(int, QRectF)
    zone_updated_signal = pyqtSignal(int, QRectF)
    prs_updated_signal = pyqtSignal(str, QRectF)
    keypoint_added_signal = pyqtSignal(object, int, int)

    def __init__(self):
        super().__init__()
        self.num_pose_keypoints = 2
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.scene.selectionChanged.connect(self.on_selection_changed)

        self.image_item = None
        self.drawing = False
        self.start_pt = None
        self.current_rect_item = None
        
        # Thêm biến cho vẽ Polygon
        self.drawing_poly = False
        self.poly_points = []
        self.temp_poly_item = None

        self.current_tool = "DEFECT_ZONE"  # DEFECT_ZONE, SEARCH1, PATTERN1, SEARCH2, PATTERN2, YOLO_BBOX, YOLO_POLY
        self.current_yolo_class_id = 0
        self.zone_items = []  # Defect zone items (thay cho roi_items)
        self.prs_items = {}   # Search và Pattern items
        self.yolo_items = []
        self._pan_start_x = 0
        self._pan_start_y = 0
        self.current_cv_img = None
        self.auto_seg_otsu = True
        self.auto_seg_invert = True
        self.auto_seg_th = 128

    def load_image(self, filepath):
        """Load ảnh từ file path, xóa tất cả items cũ"""
        self.scene.clear()
        self.zone_items.clear()
        self.prs_items.clear()
        self.yolo_items.clear()
        img = cv2_imread(filepath)
        if img is None:
            return False
        self.current_cv_img = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixmap = numpy_to_qpixmap(img_rgb)
        self.image_item = self.scene.addPixmap(pixmap)
        h, w = img.shape[:2]
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return True

    def load_image_from_array(self, cv_img):
        """Load ảnh từ numpy array (BGR hoặc Grayscale), xóa tất cả items cũ"""
        self.scene.clear()
        self.zone_items.clear()
        self.prs_items.clear()
        self.yolo_items.clear()
        if cv_img is None:
            self.current_cv_img = None
            return
        self.current_cv_img = cv_img.copy()
        if len(cv_img.shape) == 2:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pixmap = numpy_to_qpixmap(img_rgb)
        self.image_item = self.scene.addPixmap(pixmap)
        h, w = cv_img.shape[:2]
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def display_result_image(self, cv_img):
        """Hiển thị ảnh kết quả (không vẽ annotation)"""
        if cv_img is None:
            return
        if len(cv_img.shape) == 2:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pixmap = numpy_to_qpixmap(img_rgb)
        self.scene.clear()
        self.zone_items.clear()
        self.prs_items.clear()
        self.yolo_items.clear()
        self.image_item = self.scene.addPixmap(pixmap)
        self.current_cv_img = cv_img.copy()
        h, w = cv_img.shape[:2]
        self.setSceneRect(0, 0, w, h)

    # --- ZOOM & PAN ---
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self.scene.update()

    def mousePressEvent(self, event):
        # Right-click = Pan
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_start_x = event.pos().x()
            self._pan_start_y = event.pos().y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Left-click = Chọn item hoặc vẽ mới
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            # Nếu click trúng handle của item đang chọn hoặc click item trong chế độ NONE -> cho phép kéo/resize
            if isinstance(item, (ResizableRectItem, ResizablePolygonItem)):
                local_pt = item.mapFromScene(self.mapToScene(event.pos()))
                if item.isSelected() and item.get_handle_at_pos(local_pt) is not None:
                    super().mousePressEvent(event)
                    return
                if self.current_tool == "NONE":
                    super().mousePressEvent(event)
                    return

            if getattr(self, 'current_tool', '') in ("YOLO_POSE", "DEFECT_POSE_KPT"):
                target_items = [it for it in self.scene.selectedItems() 
                                if getattr(it, 'item_type', '') in ("YOLO_LABEL_POSE", "DEFECT_ZONE", "DEFECT_ZONE_POLY")]
                scene_pt = self.mapToScene(event.pos())
                if not target_items:
                    search_candidates = []
                    if hasattr(self, 'yolo_items'):
                        search_candidates.extend(self.yolo_items)
                    if hasattr(self, 'zone_items'):
                        search_candidates.extend(self.zone_items)

                    for it in search_candidates:
                        if getattr(it, 'item_type', '') in ("YOLO_LABEL_POSE", "DEFECT_ZONE", "DEFECT_ZONE_POLY"):
                            if it.contains(it.mapFromScene(scene_pt)):
                                it.setSelected(True)
                                target_items = [it]
                                break
                    if not target_items:
                        pose_boxes = [it for it in search_candidates if getattr(it, 'item_type', '') in ("YOLO_LABEL_POSE", "DEFECT_ZONE", "DEFECT_ZONE_POLY")]
                        if len(pose_boxes) == 1:
                            pose_boxes[0].setSelected(True)
                            target_items = [pose_boxes[0]]
                if target_items:
                    p_item = target_items[0]
                    num_kpts = getattr(self, 'num_pose_keypoints', 2)
                    if not hasattr(p_item, 'keypoints') or p_item.keypoints is None:
                        p_item.keypoints = []
                    if len(p_item.keypoints) < num_kpts:
                        click_pt = p_item.mapFromScene(scene_pt)
                        v = 1 if event.modifiers() == Qt.KeyboardModifier.ShiftModifier else 2
                        p_item.keypoints.append({'pos': click_pt, 'v': v})
                        p_item.update()
                        if hasattr(self, 'keypoint_added_signal'):
                            self.keypoint_added_signal.emit(p_item, len(p_item.keypoints), num_kpts)
                        event.accept()
                        return

            if self.image_item:
                if self.current_tool in ("DEFECT_ZONE_POLY", "IGNORE_ZONE_POLY", "YOLO_POLY"):
                    click_pt = self.mapToScene(event.pos())
                    self.drawing_poly = True
                    self.poly_points.append(click_pt)
                    
                    if self.temp_poly_item is None:
                        self.temp_poly_item = QGraphicsPolygonItem()
                        if "IGNORE" in self.current_tool:
                            self.temp_poly_item.setPen(QPen(QColor(245, 60, 60), 2, Qt.PenStyle.DashDotLine))
                            self.temp_poly_item.setBrush(QBrush(QColor(245, 60, 60, 45)))
                        elif "DEFECT" in self.current_tool:
                            self.temp_poly_item.setPen(QPen(QColor(255, 140, 0), 2, Qt.PenStyle.DashLine))
                            self.temp_poly_item.setBrush(QBrush(QColor(255, 100, 0, 35)))
                        else:
                            self.temp_poly_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine))
                            self.temp_poly_item.setBrush(QBrush(QColor(0, 255, 0, 35)))
                        self.scene.addItem(self.temp_poly_item)
                        
                    poly = QPolygonF(self.poly_points)
                    self.temp_poly_item.setPolygon(poly)
                    event.accept()
                    return
                elif self.current_tool == "NONE":
                    super().mousePressEvent(event)
                    return
                else:
                    self.drawing = True
                    self.start_pt = self.mapToScene(event.pos())
                    self.current_rect_item = QGraphicsRectItem(QRectF(self.start_pt, self.start_pt))

                    # Màu sắc cho nét vẽ nháp
                    if "SEARCH" in self.current_tool:
                        self.current_rect_item.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
                    elif "PATTERN" in self.current_tool:
                        self.current_rect_item.setPen(QPen(QColor(0, 255, 255), 2))
                    elif "IGNORE" in self.current_tool:
                        self.current_rect_item.setPen(QPen(QColor(245, 60, 60), 2, Qt.PenStyle.DashDotLine))
                    elif "YOLO" in self.current_tool:
                        self.current_rect_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine))
                    else:
                        # Defect Zone - cam dashed
                        self.current_rect_item.setPen(QPen(QColor(255, 140, 0), 2, Qt.PenStyle.DashLine))

                    self.scene.addItem(self.current_rect_item)
                    
                    # Hủy chọn mọi item khác để không kéo nhầm
                    for it in self.scene.selectedItems():
                        it.setSelected(False)
                        
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Pan
        if event.buttons() == Qt.MouseButton.RightButton:
            dx = event.pos().x() - self._pan_start_x
            dy = event.pos().y() - self._pan_start_y
            self._pan_start_x = event.pos().x()
            self._pan_start_y = event.pos().y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
            event.accept()
            return

        # Drawing Poly
        if self.drawing_poly and self.temp_poly_item:
            current_pt = self.mapToScene(event.pos())
            pts = self.poly_points.copy()
            pts.append(current_pt)
            self.temp_poly_item.setPolygon(QPolygonF(pts))
            event.accept()
            return

        # Drawing Rect
        if self.drawing and self.current_rect_item:
            current_pt = self.mapToScene(event.pos())
            rect = QRectF(self.start_pt, current_pt).normalized()
            self.current_rect_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing_poly:
            # Kết thúc vẽ Polygon
            if len(self.poly_points) > 2:
                final_poly = QPolygonF(self.poly_points)
                poly_item = None
                if self.current_tool == "DEFECT_ZONE_POLY":
                    poly_item = ResizablePolygonItem(final_poly, self, "DEFECT_ZONE_POLY")
                    self.scene.addItem(poly_item)
                    self.zone_items.append(poly_item)
                    self.zone_added_signal.emit(poly_item.boundingRect())
                elif self.current_tool == "IGNORE_ZONE_POLY":
                    poly_item = ResizablePolygonItem(final_poly, self, "IGNORE_ZONE_POLY")
                    self.scene.addItem(poly_item)
                    self.zone_items.append(poly_item)
                    self.zone_added_signal.emit(poly_item.boundingRect())
                elif self.current_tool == "YOLO_POLY":
                    poly_item = ResizablePolygonItem(final_poly, self, "YOLO_LABEL_POLY", self.current_yolo_class_id)
                    self.scene.addItem(poly_item)
                    self.yolo_items.append(poly_item)
                    
                if poly_item is not None:
                    poly_item.setSelected(True)

            # Cleanup
            if self.temp_poly_item:
                self.scene.removeItem(self.temp_poly_item)
                self.temp_poly_item = None
            self.poly_points = []
            self.drawing_poly = False
            event.accept()
            return
            
        super().mouseDoubleClickEvent(event)

    def apply_auto_seg(self, rect, target_item=None):
        """Áp dụng binary thresholding lên vùng rect và trích xuất đường bao đa giác"""
        if getattr(self, 'current_cv_img', None) is None:
            return None
        
        img_h, img_w = self.current_cv_img.shape[:2]
        rx = max(0, int(rect.x()))
        ry = max(0, int(rect.y()))
        rw = min(img_w - rx, int(rect.width()))
        rh = min(img_h - ry, int(rect.height()))
        
        if rw <= 3 or rh <= 3:
            return None
            
        patch = self.current_cv_img[ry:ry+rh, rx:rx+rw]
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch.copy()
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        use_otsu = getattr(self, 'auto_seg_otsu', False)
        invert = getattr(self, 'auto_seg_invert', True)
        th_val = getattr(self, 'auto_seg_th', 128)
        
        thresh_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if use_otsu:
            thresh_mode |= cv2.THRESH_OTSU
            _, bin_img = cv2.threshold(blurred, 0, 255, thresh_mode)
        else:
            _, bin_img = cv2.threshold(blurred, th_val, 255, thresh_mode)
        
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_poly = None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) >= 3:
                epsilon = max(1.0, 0.006 * cv2.arcLength(cnt, True))
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    pts = [QPointF(float(rx + p[0][0]), float(ry + p[0][1])) for p in approx]
                    final_poly = QPolygonF(pts)
        
        if final_poly is None:
            pts = [
                QPointF(rect.left(), rect.top()),
                QPointF(rect.right(), rect.top()),
                QPointF(rect.right(), rect.bottom()),
                QPointF(rect.left(), rect.bottom())
            ]
            final_poly = QPolygonF(pts)
            
        if target_item is not None and target_item in self.yolo_items and target_item.scene() == self.scene:
            target_item.setPolygon(final_poly)
            target_item.update()
            return target_item
        else:
            poly_item = ResizablePolygonItem(final_poly, self, "YOLO_LABEL_POLY", self.current_yolo_class_id)
            self.scene.addItem(poly_item)
            self.yolo_items.append(poly_item)
            poly_item.setSelected(True)
            self.last_auto_seg_item = poly_item
            self.last_auto_seg_rect = rect
            return poly_item

    def reapply_last_auto_seg(self):
        """Khi người dùng kéo slider hoặc chỉnh threshold, cập nhật đường bao realtime!"""
        if getattr(self, 'last_auto_seg_rect', None) is not None and getattr(self, 'last_auto_seg_item', None) is not None:
            if self.last_auto_seg_item in self.yolo_items and self.last_auto_seg_item.scene() == self.scene:
                self.apply_auto_seg(self.last_auto_seg_rect, target_item=self.last_auto_seg_item)

    def mouseReleaseEvent(self, event):
        # Pan end
        if event.button() == Qt.MouseButton.RightButton:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        # Drawing end
        if event.button() == Qt.MouseButton.LeftButton and self.drawing and self.current_rect_item:
            rect = self.current_rect_item.rect()
            self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None
            self.drawing = False

            if rect.width() > 10 and rect.height() > 10:
                if self.current_tool == "YOLO_BBOX":
                    final_item = ResizableRectItem(rect, self, "YOLO_LABEL", self.current_yolo_class_id)
                    self.scene.addItem(final_item)
                    self.yolo_items.append(final_item)
                elif self.current_tool == "YOLO_POSE":
                    final_item = ResizableRectItem(rect, self, "YOLO_LABEL_POSE", self.current_yolo_class_id)
                    self.scene.addItem(final_item)
                    self.yolo_items.append(final_item)
                elif self.current_tool == "YOLO_AUTO_SEG":
                    self.apply_auto_seg(rect)
                    event.accept()
                    return
                elif self.current_tool in ("DEFECT_ZONE", "IGNORE_ZONE"):
                    final_item = ResizableRectItem(rect, self, self.current_tool)
                    self.scene.addItem(final_item)
                    self.zone_items.append(final_item)
                    self.zone_added_signal.emit(rect)
                else:
                    if self.current_tool in self.prs_items:
                        old_item = self.prs_items[self.current_tool]
                        if old_item is not None and old_item.scene() == self.scene:
                            old_item.setSelected(False)
                            self.scene.removeItem(old_item)
                    final_item = ResizableRectItem(rect, self, self.current_tool)
                    self.scene.addItem(final_item)
                    self.prs_items[self.current_tool] = final_item
                    self.on_prs_geometry_changed(final_item)
                final_item.setSelected(True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # --- SIGNALS & CALLBACKS ---
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            deleted_any = False
            for item in self.scene.selectedItems():
                if hasattr(self, 'yolo_items') and item in self.yolo_items:
                    self.scene.removeItem(item)
                    self.yolo_items.remove(item)
                    deleted_any = True
            if deleted_any:
                event.accept()
                return
            if hasattr(self, 'on_delete_unselected') and callable(self.on_delete_unselected):
                self.on_delete_unselected()
                event.accept()
                return
        elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Z):
            tool = getattr(self, 'current_tool', '')
            if tool in ("YOLO_POSE", "DEFECT_POSE_KPT"):
                for item in self.scene.selectedItems():
                    if getattr(item, 'item_type', '') in ("YOLO_LABEL_POSE", "DEFECT_ZONE", "DEFECT_ZONE_POLY"):
                        if hasattr(item, 'keypoints') and item.keypoints:
                            item.keypoints.pop()
                            item.update()
                            if hasattr(self, 'keypoint_added_signal'):
                                num_kpts = getattr(self, 'num_pose_keypoints', 2)
                                self.keypoint_added_signal.emit(item, len(item.keypoints), num_kpts)
        super().keyPressEvent(event)

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        if not items:
            return
        item = items[0]
        if isinstance(item, (ResizableRectItem, ResizablePolygonItem)) and item.item_type in ["DEFECT_ZONE", "DEFECT_ZONE_POLY", "IGNORE_ZONE", "IGNORE_ZONE_POLY"]:
            try:
                idx = self.zone_items.index(item)
                if hasattr(item, 'polygon'):
                    poly_r = item.polygon().boundingRect()
                    pos = item.pos()
                    r = QRectF(poly_r.x() + pos.x(), poly_r.y() + pos.y(), poly_r.width(), poly_r.height())
                else:
                    rect = item.rect()
                    pos = item.pos()
                    r = QRectF(rect.x() + pos.x(), rect.y() + pos.y(), rect.width(), rect.height())
                self.zone_selected_signal.emit(idx, r)
            except ValueError:
                pass

    def on_zone_geometry_changed(self, item):
        try:
            idx = self.zone_items.index(item)
            if hasattr(item, 'polygon'):
                poly_r = item.polygon().boundingRect()
                pos = item.pos()
                r = QRectF(poly_r.x() + pos.x(), poly_r.y() + pos.y(), poly_r.width(), poly_r.height())
            else:
                rect = item.rect()
                pos = item.pos()
                r = QRectF(rect.x() + pos.x(), rect.y() + pos.y(), rect.width(), rect.height())
            self.zone_updated_signal.emit(idx, r)
        except ValueError:
            pass

    def on_prs_geometry_changed(self, item):
        try:
            r = item.rect()
            p = item.pos()
            abs_rect = QRectF(r.x() + p.x(), r.y() + p.y(), r.width(), r.height())
            self.prs_updated_signal.emit(item.item_type, abs_rect)
        except Exception:
            pass


    # --- QUẢN LÝ DEFECT ZONES ---
    def update_zone_geometry(self, idx, x, y, w, h):
        if 0 <= idx < len(self.zone_items):
            item = self.zone_items[idx]
            item.setPos(0, 0)
            if hasattr(item, 'setRect'):
                item.setRect(QRectF(x, y, w, h))

    def remove_zone(self, idx):
        if 0 <= idx < len(self.zone_items):
            item = self.zone_items.pop(idx)
            self.scene.removeItem(item)

    def clear_all(self):
        for item in self.zone_items:
            self.scene.removeItem(item)
        for item in self.prs_items.values():
            self.scene.removeItem(item)
        for item in self.yolo_items:
            self.scene.removeItem(item)
        self.zone_items.clear()
        self.prs_items.clear()
        self.yolo_items.clear()

    def clear_zones_only(self):
        """Chỉ xóa Defect Zones, giữ PRS"""
        for item in self.zone_items:
            self.scene.removeItem(item)
        self.zone_items.clear()

    # --- KHÔI PHỤC TỪ RECIPE ---
    def draw_zones_from_list(self, zones):
        """Vẽ lại Defect Zones và Ignore Zones từ danh sách recipe"""
        for item in self.zone_items:
            self.scene.removeItem(item)
        self.zone_items.clear()
        for zone_data in zones:
            is_ignore = False
            if isinstance(zone_data, dict):
                is_ignore = zone_data.get("is_ignore", False) or "ignore" in zone_data.get("type", "").lower()
            if isinstance(zone_data, dict) and "polygon" in zone_data.get("type", "").lower():
                from PyQt6.QtCore import QPointF
                from PyQt6.QtGui import QPolygonF
                pts = [QPointF(p[0], p[1]) for p in zone_data["points"]]
                item_type = "IGNORE_ZONE_POLY" if is_ignore else "DEFECT_ZONE_POLY"
                zone_item = ResizablePolygonItem(QPolygonF(pts), self, item_type)
            else:
                rect = zone_data['rect'] if isinstance(zone_data, dict) else zone_data
                x, y, w, h = rect
                item_type = "IGNORE_ZONE" if is_ignore else "DEFECT_ZONE"
                zone_item = ResizableRectItem(QRectF(x, y, w, h), self, item_type)
            if isinstance(zone_data, dict) and 'keypoints' in zone_data and zone_data['keypoints']:
                from PyQt6.QtCore import QPointF
                zone_item.keypoints = []
                for kp in zone_data['keypoints']:
                    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                        kx, ky = kp[0], kp[1]
                        v = kp[2] if len(kp) > 2 else 2
                    elif isinstance(kp, dict):
                        pos = kp.get('pos', [0, 0])
                        kx, ky = pos[0], pos[1]
                        v = kp.get('v', 2)
                    else:
                        continue
                    local_pt = zone_item.mapFromScene(QPointF(float(kx), float(ky)))
                    zone_item.keypoints.append({'pos': local_pt, 'v': int(v)})
            self.scene.addItem(zone_item)
            self.zone_items.append(zone_item)

    def draw_prs_from_dict(self, tool_name, rect):
        """Vẽ lại vùng PRS từ recipe"""
        if tool_name in self.prs_items:
            old_item = self.prs_items[tool_name]
            if old_item is not None and old_item.scene() == self.scene:
                old_item.setSelected(False)
                self.scene.removeItem(old_item)
        item_type = tool_name
        new_item = ResizableRectItem(QRectF(*rect), self, item_type)
        self.scene.addItem(new_item)
        self.prs_items[tool_name] = new_item
        
    def draw_yolo_labels(self, labels, img_w, img_h):
        """Vẽ lại Labels (BBox hoặc Polygon) từ file YOLO"""
        for item in self.yolo_items:
            self.scene.removeItem(item)
        self.yolo_items.clear()
        
        for lbl in labels:
            data = lbl['data']
            class_id = int(data[0])
            if lbl['type'] == 'bbox':
                x_c, y_c, w_n, h_n = data[1:5]
                w = w_n * img_w
                h = h_n * img_h
                x = (x_c * img_w) - (w / 2)
                y = (y_c * img_h) - (h / 2)
                
                yolo_item = ResizableRectItem(QRectF(x, y, w, h), self, "YOLO_LABEL", class_id=class_id)
                self.scene.addItem(yolo_item)
                self.yolo_items.append(yolo_item)
            elif lbl['type'] == 'pose':
                x_c, y_c, w_n, h_n = data[1:5]
                w = w_n * img_w
                h = h_n * img_h
                x = (x_c * img_w) - (w / 2)
                y = (y_c * img_h) - (h / 2)
                
                yolo_item = ResizableRectItem(QRectF(x, y, w, h), self, "YOLO_LABEL_POSE", class_id=class_id)
                self.scene.addItem(yolo_item)
                self.yolo_items.append(yolo_item)
                
                yolo_item.keypoints = []
                for i in range(5, len(data), 3):
                    if i + 2 < len(data):
                        kpt_x = data[i] * img_w
                        kpt_y = data[i+1] * img_h
                        v = int(data[i+2])
                        scene_pt = QPointF(kpt_x, kpt_y)
                        local_pt = yolo_item.mapFromScene(scene_pt)
                        yolo_item.keypoints.append({'pos': local_pt, 'v': v})
            else:
                poly = QPolygonF()
                for i in range(1, len(data), 2):
                    if i + 1 < len(data):
                        poly.append(QPointF(data[i] * img_w, data[i+1] * img_h))
                
                yolo_item = ResizablePolygonItem(poly, self, "YOLO_LABEL_POLY", class_id=class_id)
                self.scene.addItem(yolo_item)
                self.yolo_items.append(yolo_item)

    def get_item_rect(self, tool_name):
        """Lấy tọa độ [x, y, w, h] của item PRS theo tên tool"""
        if tool_name in self.prs_items:
            item = self.prs_items[tool_name]
            r = item.rect()
            p = item.pos()
            return [int(r.x() + p.x()), int(r.y() + p.y()), int(r.width()), int(r.height())]
        return None


# ==========================================
# WIDGET PREVIEW DEFECT ĐÃ TẠO
# ==========================================
class DefectPreviewWidget(QFrame):
    """Widget hiển thị preview ảnh defect đã tạo"""
    def __init__(self, title="Preview", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #555; border-radius: 4px; background: #2a2a2a; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.label_title = QLabel(title)
        self.label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_title.setStyleSheet("font-weight: bold; color: #ddd; border: none;")

        self.label_img = QLabel("No Image")
        self.label_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_img.setFixedSize(180, 180)
        self.label_img.setStyleSheet("border: 1px dashed #666; background: #1a1a1a; color: #888;")

        self.label_info = QLabel("---")
        self.label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_info.setStyleSheet("font-size: 10px; color: #aaa; border: none;")

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_img)
        layout.addWidget(self.label_info)

    def update_preview(self, cv_img, info_text=""):
        """Cập nhật ảnh preview và thông tin"""
        if cv_img is not None:
            if len(cv_img.shape) == 2:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pixmap = numpy_to_qpixmap(img_rgb)
            self.label_img.setPixmap(pixmap.scaled(self.label_img.size(),
                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))
        if info_text:
            self.label_info.setText(info_text)
            self.setStyleSheet("QFrame { border: 2px solid #f5a623; border-radius: 4px; background: #2e2a1b; }")

    def reset_style(self):
        """Reset về style mặc định"""
        self.label_info.setText("---")
        self.setStyleSheet("QFrame { border: 1px solid #555; border-radius: 4px; background: #2a2a2a; }")
        self.label_img.clear()
        self.label_img.setText("No Image")
