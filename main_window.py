"""
Defect & Dataset Studio AI — Synthetic Defect Generator & YOLO Dataset Manager.
6 Tabs:
  1. Recipe & Zone Setup — Master, PRS 1 & 2, Defect Zones, Keypoints
  2. Batch Generate Defects — Sinh lỗi nhân tạo, xuất YOLO Detect, Seg, Pose
  3. Manual YOLO Labeling — Gán nhãn & chỉnh sửa thủ công BBox, Seg Polygon, Pose
  4. Auto PRS & Labeling (Seg / Pose) — Căn chỉnh PRS tự động & gán nhãn
  5. Batch Crop Zones — Cắt trích xuất hàng loạt các vùng linh kiện/lỗi
  6. Dataset Manager — Quản lý, thống kê, lọc, ghép và chia tập dữ liệu YOLO
"""

import os
import glob
import json
import cv2
import numpy as np
import random
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QFileDialog, QTabWidget, QLabel,
                              QLineEdit, QGroupBox, QGridLayout, QListWidget,
                              QRadioButton, QButtonGroup, QCheckBox, QComboBox,
                              QMessageBox, QTextEdit, QScrollArea, QSpinBox,
                              QDoubleSpinBox, QSplitter, QProgressBar, QSlider,
                              QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
                              QListWidgetItem, QInputDialog, QDialog, QAbstractItemView)
from PyQt6.QtCore import Qt, QSettings, QPointF, QRectF
from PyQt6.QtGui import QFont, QTextCursor, QPolygonF, QColor, QPen, QBrush

from custom_widgets import ImageViewer, DefectPreviewWidget, ResizablePolygonItem, ResizableRectItem
from worker import (DefectWorkerThread, AutoPrsSegWorkerThread, BatchCropWorkerThread,
                    DatasetScanWorkerThread, DatasetBuilderWorkerThread)
from brush_engine import BrushEngine
from utils import img_to_b64, b64_to_img, numpy_to_qpixmap, cv2_imread, cv2_imwrite, parse_yolo_line


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect & Dataset Studio AI — Synthetic Defect & YOLO Dataset Manager")
        self.resize(1400, 900)

        # Dữ liệu Recipe
        self.recipe_data = {
            "master_image_b64": "",
            "prs1": {},
            "prs2": {},
            "defect_zones": []
        }
        self.current_setup_img_gray = None
        self.master_image_path = ""
        self.worker = None
        self.prs_worker = None
        self.crop_worker = None
        self.dataset_scan_worker = None
        self.dataset_builder_worker = None
        self.current_prs_input_folder = ""
        self.current_prs_output_folder = ""
        self.current_prs_img_path = ""
        self.current_crop_input_folder = ""
        self.current_crop_output_folder = ""
        
        # Dataset Manager state
        self.dataset_source_paths = []
        self.scanned_dataset_data = {}
        self.current_filtered_images = []
        self.builder_source_rows = []
        
        self.ui_is_updating = False

        self._apply_dark_theme()
        self.init_ui()
        self.load_settings()

    def _apply_dark_theme(self):
        """Áp dụng dark theme cho toàn bộ ứng dụng"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 12px; }
            QTabWidget::pane { border: 1px solid #45475a; background: #1e1e2e; }
            QTabBar::tab { background: #313244; color: #a6adc8; padding: 8px 18px; margin-right: 2px;
                           border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #45475a; color: #f5a623; font-weight: bold; }
            QTabBar::tab:hover { background: #585b70; }
            QGroupBox { border: 1px solid #585b70; border-radius: 8px; margin-top: 14px;
                        padding-top: 20px; font-weight: bold; color: #cdd6f4; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px;
                               color: #f5a623; }
            QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #585b70;
                          border-radius: 6px; padding: 7px 14px; min-height: 26px; }
            QPushButton:hover { background-color: #45475a; border-color: #f5a623; }
            QPushButton:pressed { background-color: #1e1e2e; }
            QLineEdit, QComboBox {
                background-color: #313244; color: #cdd6f4; border: 1px solid #585b70;
                border-radius: 4px; padding: 5px; }
            QSpinBox, QDoubleSpinBox {
                background-color: #313244; color: #cdd6f4; border: 1px solid #585b70;
                border-radius: 4px; padding: 2px; min-height: 24px; }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                width: 16px; border-left: 1px solid #585b70; background: #45475a; 
                border-top-right-radius: 3px; }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                width: 16px; border-left: 1px solid #585b70; background: #45475a; 
                border-bottom-right-radius: 3px; }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, 
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: #585b70; }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #f5a623; }
            QListWidget { background-color: #313244; color: #cdd6f4; border: 1px solid #585b70;
                          border-radius: 4px; }
            QListWidget::item:selected { background-color: #f5a623; color: #1e1e2e; }
            QTextEdit { background-color: #11111b; color: #a6e3a1; border: 1px solid #313244;
                        font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px;
                        border-radius: 4px; }
            QRadioButton { color: #cdd6f4; spacing: 5px; }
            QRadioButton::indicator:checked { background-color: #f5a623; border: 2px solid #f5a623; border-radius: 7px; }
            QCheckBox { color: #cdd6f4; spacing: 5px; }
            QCheckBox::indicator:checked { background-color: #f5a623; border: 1px solid #f5a623; border-radius: 3px; }
            QLabel { color: #bac2de; }
            QProgressBar { background-color: #313244; border: 1px solid #585b70; border-radius: 4px;
                           text-align: center; color: #1e1e2e; font-weight: bold; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #f5a623, stop:1 #fab387); border-radius: 3px; }
            QScrollArea { border: none; background: transparent; }
            QSplitter::handle { background-color: #45475a; }
            QSlider::groove:horizontal { background: #313244; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f5a623; width: 16px; height: 16px;
                                         margin: -5px 0; border-radius: 8px; }
            QSlider::sub-page:horizontal { background: #f5a623; border-radius: 3px; }
        """)

    # ==========================================
    # INIT UI — 2 TABS
    # ==========================================
    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_recipe = QWidget()
        self.setup_tab_recipe()
        self.tabs.addTab(self.tab_recipe, "📋 1. Recipe & Zone Setup")

        self.tab_batch = QWidget()
        self.setup_tab_batch()
        self.tabs.addTab(self.tab_batch, "🚀 2. Batch Generate Defects")

        self.tab_viewer = QWidget()
        self.setup_tab_viewer()
        self.tabs.addTab(self.tab_viewer, "✍ 3. Manual YOLO Labeling")

        self.tab_auto_prs = QWidget()
        self.setup_tab_auto_prs()
        self.tabs.addTab(self.tab_auto_prs, "🎯 4. Auto PRS & Labeling (Seg / Pose)")

        self.tab_crop = QWidget()
        self.setup_tab_crop()
        self.tabs.addTab(self.tab_crop, "✂ 5. Batch Crop Zones")

        self.tab_dataset = QWidget()
        self.setup_tab_dataset()
        self.tabs.addTab(self.tab_dataset, "📊 6. Dataset Manager")

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        if index == 3:  # Tab 4: Auto PRS & YOLO-Seg
            self.refresh_auto_prs_tab()
        elif index == 4:  # Tab 5: Batch Crop Zones
            self.refresh_crop_tab()
        elif index == 5:  # Tab 6: Dataset Manager
            self.refresh_dataset_tab()

    # ==========================================
    # TAB 1: RECIPE & ZONE SETUP
    # ==========================================
    def setup_tab_recipe(self):
        layout = QHBoxLayout()

        # === BÊN TRÁI: Viewer + toolbar ===
        left_layout = QVBoxLayout()

        # Toolbar vẽ
        tool_layout = QHBoxLayout()
        self.tool_group = QButtonGroup(self)

        btn_select = QRadioButton("🖱 Chọn / Sửa")
        btn_zone = QRadioButton("🟧 Defect Zone")
        btn_zone.setChecked(True)
        btn_poly = QRadioButton("🔺 Poly Zone")
        btn_ignore = QRadioButton("🚫 Ignore Zone")
        btn_ignore_poly = QRadioButton("⛔ Poly Ignore")
        btn_s1 = QRadioButton("🟨 Search 1")
        btn_p1 = QRadioButton("🟦 Pattern 1")
        btn_s2 = QRadioButton("🟨 Search 2")
        btn_p2 = QRadioButton("🟦 Pattern 2")
        self.btn_tool_kpt = QRadioButton("📍 Chấm điểm Pose")

        for idx, btn in enumerate([btn_select, btn_zone, btn_poly, btn_ignore, btn_ignore_poly, btn_s1, btn_p1, btn_s2, btn_p2, self.btn_tool_kpt]):
            self.tool_group.addButton(btn, idx)
            tool_layout.addWidget(btn)

        self.tool_group.idClicked.connect(self.on_tool_changed)
        left_layout.addLayout(tool_layout)

        # Image Viewer
        self.viewer = ImageViewer()
        self.viewer.zone_added_signal.connect(self.on_zone_added_from_viewer)
        self.viewer.zone_selected_signal.connect(self.on_zone_selected_from_viewer)
        self.viewer.zone_updated_signal.connect(self.on_zone_updated_from_viewer)
        self.viewer.prs_updated_signal.connect(self.on_prs_updated_from_viewer)
        self.viewer.keypoint_added_signal.connect(self.on_tab1_keypoint_added)
        left_layout.addWidget(self.viewer)

        # Nút load ảnh master
        btn_load_img = QPushButton("📷 Nạp Ảnh Master (Setup Image)")
        btn_load_img.setMinimumHeight(40)
        btn_load_img.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #1565C0, stop:1 #1976D2);
                          color: white; font-weight: bold; font-size: 13px;
                          border: none; border-radius: 8px; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                stop:0 #1976D2, stop:1 #42A5F5); }
        """)
        btn_load_img.clicked.connect(self.load_setup_image)
        left_layout.addWidget(btn_load_img)

        layout.addLayout(left_layout, stretch=5)

        # === BÊN PHẢI: Control panels ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 4, 4)

        # --- PRS GROUP ---
        grp_prs = QGroupBox("🎯 PRS Setup (Học Pattern)")
        lay_prs = QVBoxLayout()

        self.btn_learn1 = QPushButton("📐 Learn PRS 1")
        self.btn_learn1.clicked.connect(lambda: self.learn_prs(1))
        lay_prs.addWidget(self.btn_learn1)

        lay_th1 = QHBoxLayout()
        lay_th1.addWidget(QLabel("Threshold PRS 1:"))
        self.txt_prs1_th = QDoubleSpinBox()
        self.txt_prs1_th.setRange(0, 1.0)
        self.txt_prs1_th.setSingleStep(0.05)
        self.txt_prs1_th.setValue(0.7)
        self.txt_prs1_th.setDecimals(2)
        lay_th1.addWidget(self.txt_prs1_th)
        lay_prs.addLayout(lay_th1)

        self.btn_learn2 = QPushButton("📐 Learn PRS 2")
        self.btn_learn2.clicked.connect(lambda: self.learn_prs(2))
        lay_prs.addWidget(self.btn_learn2)

        lay_th2 = QHBoxLayout()
        lay_th2.addWidget(QLabel("Threshold PRS 2:"))
        self.txt_prs2_th = QDoubleSpinBox()
        self.txt_prs2_th.setRange(0, 1.0)
        self.txt_prs2_th.setSingleStep(0.05)
        self.txt_prs2_th.setValue(0.7)
        self.txt_prs2_th.setDecimals(2)
        lay_th2.addWidget(self.txt_prs2_th)
        lay_prs.addLayout(lay_th2)

        self.set_prs_button_state(1, False)
        self.set_prs_button_state(2, False)

        grp_prs.setLayout(lay_prs)
        right_layout.addWidget(grp_prs)

        # --- DEFECT ZONE GROUP ---
        grp_zone = QGroupBox("🟧 Defect Zones (Vùng thêm lỗi)")
        lay_zone = QVBoxLayout()

        lay_coords = QHBoxLayout()
        for label_text, obj_name in zip(["X:", "Y:", "W:", "H:"],
                                         ["txt_zx", "txt_zy", "txt_zw", "txt_zh"]):
            lay_coords.addWidget(QLabel(label_text))
            txt = QLineEdit()
            txt.setMaximumWidth(60)
            txt.textChanged.connect(self.on_zone_input_changed)
            setattr(self, obj_name, txt)
            lay_coords.addWidget(txt)
        lay_zone.addLayout(lay_coords)

        self.list_zones = QListWidget()
        self.list_zones.currentRowChanged.connect(self.on_list_zone_selected)
        lay_zone.addWidget(self.list_zones)

        lay_zone_btn = QHBoxLayout()
        btn_del_zone = QPushButton("🗑 Xóa Zone")
        btn_del_zone.clicked.connect(self.delete_selected_zone)
        btn_clear_zone = QPushButton("🗑 Xóa Tất Cả")
        btn_clear_zone.clicked.connect(self.clear_all_zones)
        btn_dup_zone = QPushButton("📋 Nhân bản")
        btn_dup_zone.setToolTip("Tạo một bản sao của Defect Zone đang chọn")
        btn_dup_zone.clicked.connect(self.duplicate_selected_zone)
        lay_zone_btn.addWidget(btn_del_zone)
        lay_zone_btn.addWidget(btn_clear_zone)
        lay_zone_btn.addWidget(btn_dup_zone)
        lay_zone.addLayout(lay_zone_btn)

        grp_zone.setLayout(lay_zone)
        right_layout.addWidget(grp_zone)

        # --- POSE KEYPOINTS GROUP (Tab 1: Chọn số điểm & Chấm điểm Master) ---
        grp_pose = QGroupBox("📍 YOLO-Pose Keypoints (Thiết lập điểm)")
        lay_pose = QVBoxLayout()

        lay_num_kpts = QHBoxLayout()
        lay_num_kpts.addWidget(QLabel("Số điểm chấm:"))
        self.spn_tab1_pose_kpts = QSpinBox()
        self.spn_tab1_pose_kpts.setRange(1, 99)
        self.spn_tab1_pose_kpts.setValue(2)
        self.spn_tab1_pose_kpts.valueChanged.connect(self.on_tab1_pose_kpts_changed)
        lay_num_kpts.addWidget(self.spn_tab1_pose_kpts)
        lay_pose.addLayout(lay_num_kpts)

        self.lbl_tab1_kpt_status = QLabel("Trạng thái: Chưa chọn Zone")
        self.lbl_tab1_kpt_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        lay_pose.addWidget(self.lbl_tab1_kpt_status)

        lay_kpt_actions = QHBoxLayout()
        self.btn_tab1_start_kpt = QPushButton("📍 Chấm điểm")
        self.btn_tab1_start_kpt.setToolTip("Kích hoạt chế độ chấm điểm trực tiếp trên Defect Zone đang chọn")
        self.btn_tab1_start_kpt.clicked.connect(self.start_tab1_pose_labeling)

        self.btn_tab1_auto_kpt = QPushButton("⚡ Tự chấm")
        self.btn_tab1_auto_kpt.setToolTip("Tự động đặt các điểm theo số lượng đã chọn (tâm, 2 đầu mút hoặc các góc)")
        self.btn_tab1_auto_kpt.clicked.connect(self.auto_tab1_place_kpts)

        self.btn_tab1_clear_kpt = QPushButton("🧹 Xóa điểm")
        self.btn_tab1_clear_kpt.setToolTip("Xóa các điểm đã chấm trên Defect Zone này")
        self.btn_tab1_clear_kpt.clicked.connect(self.clear_tab1_selected_zone_kpts)

        lay_kpt_actions.addWidget(self.btn_tab1_start_kpt)
        lay_kpt_actions.addWidget(self.btn_tab1_auto_kpt)
        lay_kpt_actions.addWidget(self.btn_tab1_clear_kpt)
        lay_pose.addLayout(lay_kpt_actions)

        lbl_kpt_hint = QLabel("💡 Mẹo: Bấm 'Chấm điểm' rồi click chuột vào Zone trên ảnh Master.\nNhấn Backspace hoặc Ctrl+Z để xóa điểm vừa chấm.\nCác điểm sẽ tự động co giãn / quay theo PRS trên Tab 4.")
        lbl_kpt_hint.setStyleSheet("color: gray; font-size: 11px;")
        lbl_kpt_hint.setWordWrap(True)
        lay_pose.addWidget(lbl_kpt_hint)

        grp_pose.setLayout(lay_pose)
        right_layout.addWidget(grp_pose)

        # --- RECIPE GROUP ---
        grp_rec = QGroupBox("💾 Recipe")
        lay_rec = QHBoxLayout()

        btn_new = QPushButton("🆕 New")
        btn_new.clicked.connect(self.new_recipe)
        btn_save = QPushButton("💾 Save")
        btn_save.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #4CAF50, stop:1 #66BB6A); color: white; font-weight: bold;"
            "border: none; border-radius: 6px; }"
            "QPushButton:hover { background: #81C784; }")
        btn_save.clicked.connect(self.save_recipe)
        btn_load = QPushButton("📂 Load")
        btn_load.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #f5a623, stop:1 #fab387); color: #1e1e2e; font-weight: bold;"
            "border: none; border-radius: 6px; }"
            "QPushButton:hover { background: #f9cb8e; }")
        btn_load.clicked.connect(self.load_recipe)

        lay_rec.addWidget(btn_new)
        lay_rec.addWidget(btn_save)
        lay_rec.addWidget(btn_load)
        grp_rec.setLayout(lay_rec)
        right_layout.addWidget(grp_rec)

        right_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        layout.addWidget(scroll, stretch=2)

        self.tab_recipe.setLayout(layout)

    # ==========================================
    # TAB 2: BATCH GENERATE DEFECTS
    # ==========================================
    def setup_tab_batch(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ==========================================
        # LEFT PANEL: Cài đặt & Điều khiển (Sidebar)
        # ==========================================
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(2, 2, 2, 2)
        left_lay.setSpacing(6)

        # Tab Widget phân nhóm cấu hình thành 2 tab gọn gàng
        self.batch_settings_tabs = QTabWidget()
        self.batch_settings_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #45475a; border-radius: 6px; }
            QTabBar::tab { padding: 6px 10px; font-weight: bold; font-size: 11px; }
            QTabBar::tab:selected { background: #313244; color: #89b4fa; border-bottom: 2px solid #89b4fa; }
        """)

        # ------------------------------------------
        # SUB-TAB 1: Nguồn ảnh, Nhãn & Bề mặt
        # ------------------------------------------
        tab_io_surface = QWidget()
        lay_io_surface = QVBoxLayout(tab_io_surface)
        lay_io_surface.setContentsMargins(4, 6, 4, 6)
        lay_io_surface.setSpacing(8)

        # 1. Input / Output
        grp_io = QGroupBox("📂 Input / Output")
        lay_io = QGridLayout()
        lay_io.setSpacing(5)

        lay_io.addWidget(QLabel("📥 Folder ảnh:"), 0, 0)
        self.txt_input_folder = QLineEdit()
        self.txt_input_folder.setPlaceholderText("Chọn folder chứa ảnh gốc...")
        self.txt_input_folder.textChanged.connect(
            lambda p: self.lbl_file_count.setText(f"{len(self._get_image_files(p))} ảnh") if (p and os.path.isdir(p)) else self.lbl_file_count.setText("0 ảnh")
        )
        lay_io.addWidget(self.txt_input_folder, 0, 1)
        btn_input = QPushButton("...")
        btn_input.setFixedWidth(36)
        btn_input.clicked.connect(self.browse_input_folder)
        lay_io.addWidget(btn_input, 0, 2)

        self.lbl_file_count = QLabel("0 ảnh")
        self.lbl_file_count.setStyleSheet("color: #f5a623; font-weight: bold;")
        lay_io.addWidget(self.lbl_file_count, 0, 3)

        lay_io.addWidget(QLabel("📤 Folder output:"), 1, 0)
        self.txt_output_folder = QLineEdit()
        self.txt_output_folder.setPlaceholderText("Chọn folder lưu ảnh defect...")
        lay_io.addWidget(self.txt_output_folder, 1, 1)
        btn_output = QPushButton("...")
        btn_output.setFixedWidth(36)
        btn_output.clicked.connect(self.browse_output_folder)
        lay_io.addWidget(btn_output, 1, 2)

        lay_io.addWidget(QLabel("📄 Định dạng:"), 2, 0)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems([".png", ".jpg", ".tif", ".bmp"])
        lay_io.addWidget(self.cmb_format, 2, 1, 1, 2)

        grp_io.setLayout(lay_io)
        lay_io_surface.addWidget(grp_io)

        # 2. YOLO Export
        grp_yolo = QGroupBox("🏷 YOLO Export")
        lay_yolo = QVBoxLayout()
        lay_yolo.setSpacing(6)

        lay_yolo_top = QHBoxLayout()
        self.chk_yolo = QCheckBox("Xuất YOLO (.txt)")
        self.chk_yolo.setChecked(True)
        self.cmb_yolo_format = QComboBox()
        self.cmb_yolo_format.addItems(["Bounding Box (v8/v11)", "Segmentation (v8-seg/11-seg)"])
        lay_yolo_top.addWidget(self.chk_yolo)
        lay_yolo_top.addWidget(self.cmb_yolo_format)
        lay_yolo.addLayout(lay_yolo_top)

        lay_yolo_folder = QHBoxLayout()
        lay_yolo_folder.addWidget(QLabel("Folder labels:"))
        self.txt_labels_folder = QLineEdit()
        self.txt_labels_folder.setPlaceholderText("Mặc định: output/labels/")
        lay_yolo_folder.addWidget(self.txt_labels_folder)
        btn_labels = QPushButton("...")
        btn_labels.setFixedWidth(36)
        btn_labels.clicked.connect(self.browse_labels_folder)
        lay_yolo_folder.addWidget(btn_labels)
        lay_yolo.addLayout(lay_yolo_folder)

        lbl_classes = QLabel("Classes: 0=scratch, 1=spot, 2=smear, 3=crack, 4=burr, 5=smudge, 6=stamp, 7=organic, 8=fiber")
        lbl_classes.setStyleSheet("color: #a6adc8; font-size: 10px;")
        lay_yolo.addWidget(lbl_classes)

        grp_yolo.setLayout(lay_yolo)
        lay_io_surface.addWidget(grp_yolo)

        # 3. Auto Surface Detection
        grp_surface = QGroupBox("🤖 Tự động phát hiện bề mặt Trắng / Đen")
        grp_surface.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay_surface = QGridLayout()
        lay_surface.setSpacing(6)

        self.chk_auto_surface = QCheckBox("✨ Bật tự động dò bề mặt (Không cần PRS / Recipe)")
        self.chk_auto_surface.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 11px;")
        lay_surface.addWidget(self.chk_auto_surface, 0, 0, 1, 4)

        lay_surface.addWidget(QLabel("Bề mặt mục tiêu:"), 1, 0)
        self.cmb_target_surface = QComboBox()
        self.cmb_target_surface.addItems([
            "bright — Chỉ vùng Sáng (White Surface)",
            "dark — Chỉ vùng Tối (Black Surface)",
            "both — Cả hai thích ứng (Dual-Surface)"
        ])
        lay_surface.addWidget(self.cmb_target_surface, 1, 1, 1, 3)

        lay_surface.addWidget(QLabel("Lùi viền:"), 2, 0)
        self.spn_surface_margin = QSpinBox()
        self.spn_surface_margin.setRange(0, 30)
        self.spn_surface_margin.setValue(4)
        self.spn_surface_margin.setSuffix(" px")
        self.spn_surface_margin.setToolTip("Erosion viền an toàn để vết lỗi không chạm mép")
        lay_surface.addWidget(self.spn_surface_margin, 2, 1)

        lay_surface.addWidget(QLabel("Diện tích min:"), 2, 2)
        self.spn_surface_min_area = QSpinBox()
        self.spn_surface_min_area.setRange(20, 50000)
        self.spn_surface_min_area.setValue(200)
        self.spn_surface_min_area.setSuffix(" px²")
        self.spn_surface_min_area.setToolTip("Bỏ qua các vụn hoặc nhiễu li ti nhỏ hơn diện tích này")
        lay_surface.addWidget(self.spn_surface_min_area, 2, 3)

        lay_surface.addWidget(QLabel("Phân ngưỡng:"), 3, 0)
        self.cmb_thresh_mode = QComboBox()
        self.cmb_thresh_mode.addItems([
            "otsu — Tự động (Otsu)",
            "manual — Thủ công"
        ])
        lay_surface.addWidget(self.cmb_thresh_mode, 3, 1)

        lay_surface.addWidget(QLabel("Ngưỡng cố định:"), 3, 2)
        self.spn_manual_thresh = QSpinBox()
        self.spn_manual_thresh.setRange(1, 254)
        self.spn_manual_thresh.setValue(128)
        self.spn_manual_thresh.setEnabled(False)
        self.cmb_thresh_mode.currentIndexChanged.connect(
            lambda idx: self.spn_manual_thresh.setEnabled(idx == 1)
        )
        lay_surface.addWidget(self.spn_manual_thresh, 3, 3)

        grp_surface.setLayout(lay_surface)
        lay_io_surface.addWidget(grp_surface)

        lay_io_surface.addStretch()

        scroll_io = QScrollArea()
        scroll_io.setWidgetResizable(True)
        scroll_io.setWidget(tab_io_surface)
        scroll_io.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.batch_settings_tabs.addTab(scroll_io, "📁 Nguồn & Nhãn")

        # ------------------------------------------
        # SUB-TAB 2: Cọ vẽ & Mẫu lỗi (Brush Engine)
        # ------------------------------------------
        tab_brush = QWidget()
        lay_brush_tab = QVBoxLayout(tab_brush)
        lay_brush_tab.setContentsMargins(4, 6, 4, 6)
        lay_brush_tab.setSpacing(6)

        # Preset bar
        lay_preset = QHBoxLayout()
        btn_save_preset = QPushButton("💾 Lưu Preset")
        btn_save_preset.clicked.connect(self.save_brush_preset)
        btn_load_preset = QPushButton("📂 Tải Preset")
        btn_load_preset.clicked.connect(self.load_brush_preset)
        lay_preset.addWidget(btn_save_preset)
        lay_preset.addWidget(btn_load_preset)
        lay_brush_tab.addLayout(lay_preset)

        # Group Brush Settings
        grp_brush = QGroupBox("🖌 Cấu hình Cọ vẽ (Brush Engine)")
        lay_brush = QGridLayout()
        lay_brush.setSpacing(5)

        # Loại Brush
        lay_brush.addWidget(QLabel("Loại cọ:"), 0, 0)
        self.chk_brushes = {}
        brush_names = [
            ("random", "🎲 Random"), ("scratch", "━ Xước"),
            ("spot", "● Đốm"), ("smear", "〰 Lem"),
            ("crack", "⚡ Nứt"), ("burr", "▓ Sần"), 
            ("smudge", "☁️ Vết ố"),
            ("stamp", "✨ Stamp"),
            ("organic", "🌿 Hạt hữu cơ"),
            ("fiber", "〰 Sợi bụi")
        ]
        brush_widget = QWidget()
        brush_grid = QGridLayout(brush_widget)
        brush_grid.setContentsMargins(0, 0, 0, 0)
        brush_grid.setSpacing(2)
        for i, (key, text) in enumerate(brush_names):
            chk = QCheckBox(text)
            if key == "random": chk.setChecked(True)
            self.chk_brushes[key] = chk
            brush_grid.addWidget(chk, i // 3, i % 3)
        lay_brush.addWidget(brush_widget, 0, 1, 1, 3)

        # Chế độ
        lay_brush.addWidget(QLabel("Chế độ hòa trộn:"), 1, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "multiply — Tự nhiên (ăn vào vân nền)",
            "darken — Làm tối (vết bẩn)",
            "lighten — Làm sáng (vết trầy)"
        ])
        lay_brush.addWidget(self.cmb_mode, 1, 1, 1, 3)

        # Separator 1
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("background-color: #585b70; max-height: 1px;")
        lay_brush.addWidget(sep1, 2, 0, 1, 4)

        # Row 3: Size
        lay_brush.addWidget(QLabel("Kích thước:"), 3, 0)
        self.spn_size_min = QSpinBox()
        self.spn_size_min.setRange(1, 100)
        self.spn_size_min.setValue(3)
        lay_brush.addWidget(self.spn_size_min, 3, 1)
        lbl_tilde1 = QLabel("~")
        lbl_tilde1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde1, 3, 2)
        self.spn_size_max = QSpinBox()
        self.spn_size_max.setRange(1, 200)
        self.spn_size_max.setValue(15)
        self.spn_size_max.setSuffix(" px")
        lay_brush.addWidget(self.spn_size_max, 3, 3)

        # Row 4: Opacity
        lay_brush.addWidget(QLabel("Độ mờ:"), 4, 0)
        self.spn_opacity_min = QDoubleSpinBox()
        self.spn_opacity_min.setRange(0.01, 1.0)
        self.spn_opacity_min.setValue(0.3)
        self.spn_opacity_min.setSingleStep(0.05)
        self.spn_opacity_min.setDecimals(2)
        lay_brush.addWidget(self.spn_opacity_min, 4, 1)
        lbl_tilde2 = QLabel("~")
        lbl_tilde2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde2, 4, 2)
        self.spn_opacity_max = QDoubleSpinBox()
        self.spn_opacity_max.setRange(0.01, 1.0)
        self.spn_opacity_max.setValue(0.9)
        self.spn_opacity_max.setSingleStep(0.05)
        self.spn_opacity_max.setDecimals(2)
        lay_brush.addWidget(self.spn_opacity_max, 4, 3)

        # Row 5: Intensity
        lay_brush.addWidget(QLabel("Cường độ:"), 5, 0)
        self.spn_int_min = QSpinBox()
        self.spn_int_min.setRange(1, 255)
        self.spn_int_min.setValue(20)
        lay_brush.addWidget(self.spn_int_min, 5, 1)
        lbl_tilde3 = QLabel("~")
        lbl_tilde3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde3, 5, 2)
        self.spn_int_max = QSpinBox()
        self.spn_int_max.setRange(1, 255)
        self.spn_int_max.setValue(80)
        lay_brush.addWidget(self.spn_int_max, 5, 3)

        # Row 6: Count
        lay_brush.addWidget(QLabel("Số lượng:"), 6, 0)
        self.spn_count_min = QSpinBox()
        self.spn_count_min.setRange(1, 50)
        self.spn_count_min.setValue(1)
        lay_brush.addWidget(self.spn_count_min, 6, 1)
        lbl_tilde4 = QLabel("~")
        lbl_tilde4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde4, 6, 2)
        self.spn_count_max = QSpinBox()
        self.spn_count_max.setRange(1, 100)
        self.spn_count_max.setValue(5)
        lay_brush.addWidget(self.spn_count_max, 6, 3)

        # Row 7: Length
        lay_brush.addWidget(QLabel("Chiều dài:"), 7, 0)
        self.spn_length_min = QSpinBox()
        self.spn_length_min.setRange(5, 1000)
        self.spn_length_min.setValue(30)
        self.spn_length_min.setSuffix(" px")
        lay_brush.addWidget(self.spn_length_min, 7, 1)
        lbl_tilde5 = QLabel("~")
        lbl_tilde5.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde5, 7, 2)
        self.spn_length_max = QSpinBox()
        self.spn_length_max.setRange(10, 2000)
        self.spn_length_max.setValue(150)
        self.spn_length_max.setSuffix(" px")
        lay_brush.addWidget(self.spn_length_max, 7, 3)

        # Row 8: Angle
        lay_brush.addWidget(QLabel("Góc xoay:"), 8, 0)
        self.spn_angle_min = QSpinBox()
        self.spn_angle_min.setRange(0, 360)
        self.spn_angle_min.setValue(0)
        self.spn_angle_min.setSuffix("°")
        lay_brush.addWidget(self.spn_angle_min, 8, 1)
        lbl_tilde6 = QLabel("~")
        lbl_tilde6.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde6, 8, 2)
        self.spn_angle_max = QSpinBox()
        self.spn_angle_max.setRange(0, 360)
        self.spn_angle_max.setValue(360)
        self.spn_angle_max.setSuffix("°")
        lay_brush.addWidget(self.spn_angle_max, 8, 3)

        # Row 9: Main Thick
        lay_brush.addWidget(QLabel("Nét chính:"), 9, 0)
        self.spn_main_thick_min = QSpinBox()
        self.spn_main_thick_min.setRange(1, 50)
        self.spn_main_thick_min.setValue(1)
        self.spn_main_thick_min.setSuffix(" px")
        lay_brush.addWidget(self.spn_main_thick_min, 9, 1)
        lbl_tilde7 = QLabel("~")
        lbl_tilde7.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde7, 9, 2)
        self.spn_main_thick_max = QSpinBox()
        self.spn_main_thick_max.setRange(1, 100)
        self.spn_main_thick_max.setValue(5)
        self.spn_main_thick_max.setSuffix(" px")
        lay_brush.addWidget(self.spn_main_thick_max, 9, 3)

        # Row 10: Sub-Branch
        lay_brush.addWidget(QLabel("Nhánh phụ:"), 10, 0)
        self.spn_sub_min = QSpinBox()
        self.spn_sub_min.setRange(0, 20)
        self.spn_sub_min.setValue(0)
        lay_brush.addWidget(self.spn_sub_min, 10, 1)
        lbl_tilde8 = QLabel("~")
        lbl_tilde8.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay_brush.addWidget(lbl_tilde8, 10, 2)
        self.spn_sub_max = QSpinBox()
        self.spn_sub_max.setRange(0, 50)
        self.spn_sub_max.setValue(3)
        lay_brush.addWidget(self.spn_sub_max, 10, 3)

        # Row 11: Sub Scale
        lay_brush.addWidget(QLabel("Tỉ lệ nhánh:"), 11, 0)
        self.spn_sub_scale = QDoubleSpinBox()
        self.spn_sub_scale.setRange(0.05, 1.0)
        self.spn_sub_scale.setValue(0.4)
        self.spn_sub_scale.setSingleStep(0.05)
        self.spn_sub_scale.setDecimals(2)
        self.spn_sub_scale.setToolTip("Tỉ lệ chiều dài nhánh phụ so với nhánh chính (0.1=ngắn, 1.0=bằng nhánh chính)")
        lay_brush.addWidget(self.spn_sub_scale, 11, 1, 1, 3)

        # Separator 2
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #585b70; max-height: 1px;")
        lay_brush.addWidget(sep2, 12, 0, 1, 4)

        # Row 13: Stamp Library
        from brush_engine import StampBank
        lbl_stamp_title = QLabel("🖼️ Stamp Mẫu thực:")
        lbl_stamp_title.setStyleSheet("font-weight: bold; color: #f5a623;")
        lay_brush.addWidget(lbl_stamp_title, 13, 0, 1, 2)
        self.lbl_stamp_count = QLabel(f"Đã nạp: {StampBank.count()}+ mẫu")
        self.lbl_stamp_count.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        lay_brush.addWidget(self.lbl_stamp_count, 13, 2, 1, 2)

        lay_brush.addWidget(QLabel("Phân bố Stamp:"), 14, 0)
        self.cmb_stamp_mode = QComboBox()
        self.cmb_stamp_mode.addItems([
            "scatter — Hạt đơn (đúng số hạt)",
            "mixed — Hỗn hợp (cụm 2-5 hạt)",
            "cluster — Cụm bụi văng"
        ])
        lay_brush.addWidget(self.cmb_stamp_mode, 14, 1, 1, 3)

        stamp_btn_box = QHBoxLayout()
        self.btn_load_custom_stamp = QPushButton("📂 Nạp thêm mẫu...")
        self.btn_load_custom_stamp.clicked.connect(self.choose_custom_stamp_image)
        self.btn_reset_stamp = QPushButton("🔄 Mặc định")
        self.btn_reset_stamp.clicked.connect(self.reset_default_stamps)
        stamp_btn_box.addWidget(self.btn_load_custom_stamp)
        stamp_btn_box.addWidget(self.btn_reset_stamp)
        lay_brush.addLayout(stamp_btn_box, 15, 0, 1, 4)

        # Row 16: Scope
        lay_brush.addWidget(QLabel("Phạm vi Count:"), 16, 0)
        self.cmb_count_scope = QComboBox()
        self.cmb_count_scope.addItems([
            "per_image — Cả ảnh (Tổng vết)",
            "per_zone — Từng vùng (Mỗi vùng Count vết)"
        ])
        lay_brush.addWidget(self.cmb_count_scope, 16, 1, 1, 3)

        grp_brush.setLayout(lay_brush)
        lay_brush_tab.addWidget(grp_brush)
        lay_brush_tab.addStretch()

        scroll_brush = QScrollArea()
        scroll_brush.setWidgetResizable(True)
        scroll_brush.setWidget(tab_brush)
        scroll_brush.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.batch_settings_tabs.addTab(scroll_brush, "🖌 Cọ vẽ & Lỗi")

        left_lay.addWidget(self.batch_settings_tabs)

        # Bottom of left sidebar: Actions & Progress
        btn_preview = QPushButton("👁 Preview Brush (Test 1 ảnh)")
        btn_preview.setMinimumHeight(36)
        btn_preview.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #7c3aed, stop:1 #a855f7); color: white; font-weight: bold;"
            "border: none; border-radius: 6px; font-size: 12px; }"
            "QPushButton:hover { background: #a78bfa; }")
        btn_preview.clicked.connect(self.preview_single)
        left_lay.addWidget(btn_preview)

        self.btn_run = QPushButton("▶ RUN ALL — Tạo Defect Hàng Loạt")
        self.btn_run.setMinimumHeight(44)
        self.btn_run.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #4CAF50, stop:1 #66BB6A);
                          color: white; font-weight: bold; font-size: 14px;
                          border: none; border-radius: 8px; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                stop:0 #66BB6A, stop:1 #81C784); }
        """)
        self.btn_run.clicked.connect(self.toggle_run)
        left_lay.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(20)
        left_lay.addWidget(self.progress_bar)

        main_splitter.addWidget(left_widget)

        # ==========================================
        # RIGHT PANEL: Preview Viewer & Log
        # ==========================================
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(2, 2, 2, 2)
        right_lay.setSpacing(4)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top area of right side: Header bar + ImageViewer
        viewer_container = QWidget()
        viewer_lay = QVBoxLayout(viewer_container)
        viewer_lay.setContentsMargins(0, 0, 0, 0)
        viewer_lay.setSpacing(4)

        viewer_header = QHBoxLayout()
        lbl_viewer_title = QLabel("🖼️ Vùng hiển thị hình ảnh Review (Zoom/Pan tự do)")
        lbl_viewer_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 13px;")
        viewer_header.addWidget(lbl_viewer_title)
        viewer_header.addStretch()

        btn_toggle_log = QPushButton("📟 Ẩn/Hiện Log")
        btn_toggle_log.setCheckable(True)
        btn_toggle_log.setChecked(True)
        btn_toggle_log.setFixedHeight(26)
        btn_toggle_log.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        viewer_header.addWidget(btn_toggle_log)
        viewer_lay.addLayout(viewer_header)

        self.batch_viewer = ImageViewer()
        viewer_lay.addWidget(self.batch_viewer)

        right_splitter.addWidget(viewer_container)

        # Bottom area of right side: Terminal Log
        log_widget = QWidget()
        log_lay = QVBoxLayout(log_widget)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(2)
        lbl_log = QLabel("📟 Terminal Log:")
        lbl_log.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        log_lay.addWidget(lbl_log)
        self.batch_log = QTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMaximumHeight(200)
        log_lay.addWidget(self.batch_log)

        right_splitter.addWidget(log_widget)

        btn_toggle_log.toggled.connect(log_widget.setVisible)

        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 1)

        right_lay.addWidget(right_splitter)
        main_splitter.addWidget(right_widget)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([700, 700])

        main_layout.addWidget(main_splitter)
        self.tab_batch.setLayout(main_layout)

    # ==========================================
    # TAB 1: CÁC HÀM XỬ LÝ
    # ==========================================
    def on_tool_changed(self, tool_id):
        tools = {
            0: "NONE",
            1: "DEFECT_ZONE",
            2: "DEFECT_ZONE_POLY",
            3: "IGNORE_ZONE",
            4: "IGNORE_ZONE_POLY",
            5: "SEARCH1",
            6: "PATTERN1",
            7: "SEARCH2",
            8: "PATTERN2",
            9: "DEFECT_POSE_KPT"
        }
        self.viewer.current_tool = tools.get(tool_id, "NONE")
        if self.viewer.current_tool == "DEFECT_POSE_KPT":
            cur_row = self.list_zones.currentRow()
            num_kpts = self.spn_tab1_pose_kpts.value() if hasattr(self, 'spn_tab1_pose_kpts') else 2
            if 0 <= cur_row < len(self.viewer.zone_items):
                item = self.viewer.zone_items[cur_row]
                item.setSelected(True)
                kpts_count = len(getattr(item, 'keypoints', []))
                self.lbl_tab1_kpt_status.setText(f"Zone {cur_row + 1}: Đang chấm {kpts_count}/{num_kpts} điểm...")
                self.statusBar().showMessage(f"📍 Chế độ chấm điểm Pose cho Zone {cur_row + 1}. Click chuột lên ảnh để chấm.", 3000)
            else:
                self.lbl_tab1_kpt_status.setText("Chế độ chấm: Hãy chọn hoặc click vào 1 Zone")

    def start_tab1_pose_labeling(self):
        """Kích hoạt công cụ chấm điểm Pose trên Tab 1"""
        if hasattr(self, 'btn_tool_kpt'):
            self.btn_tool_kpt.setChecked(True)
        self.viewer.current_tool = "DEFECT_POSE_KPT"
        cur_row = self.list_zones.currentRow()
        num_kpts = self.spn_tab1_pose_kpts.value() if hasattr(self, 'spn_tab1_pose_kpts') else 2
        if 0 <= cur_row < len(self.viewer.zone_items):
            item = self.viewer.zone_items[cur_row]
            item.setSelected(True)
            kpts_count = len(getattr(item, 'keypoints', []))
            self.lbl_tab1_kpt_status.setText(f"Zone {cur_row + 1}: Đang chấm {kpts_count}/{num_kpts} điểm...")
            self.statusBar().showMessage(f"📍 Đang ở chế độ chấm điểm Pose cho Zone {cur_row + 1}. Click chuột lên ảnh để chấm.", 4000)
        else:
            self.lbl_tab1_kpt_status.setText("Chế độ chấm: Hãy click vào một Defect Zone")
            self.statusBar().showMessage("📍 Hãy chọn hoặc click vào một Defect Zone để bắt đầu chấm điểm.", 4000)

    def clear_tab1_selected_zone_kpts(self):
        """Xóa toàn bộ điểm chấm trên Zone đang chọn"""
        cur_row = self.list_zones.currentRow()
        if 0 <= cur_row < len(self.viewer.zone_items):
            item = self.viewer.zone_items[cur_row]
            item.keypoints = []
            item.update()
            num_kpts = self.spn_tab1_pose_kpts.value() if hasattr(self, 'spn_tab1_pose_kpts') else 2
            self.lbl_tab1_kpt_status.setText(f"Zone {cur_row + 1}: Đã xóa hết điểm (0/{num_kpts})")
            self.statusBar().showMessage(f"🧹 Đã xóa điểm chấm của Zone {cur_row + 1}", 2500)
        else:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn 1 Defect Zone để xóa điểm!")

    def auto_tab1_place_kpts(self):
        """Tự động đặt các điểm Pose trên Defect Zone đang chọn theo số lượng spn_tab1_pose_kpts"""
        cur_row = self.list_zones.currentRow()
        if cur_row < 0 or cur_row >= len(self.viewer.zone_items):
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn 1 Defect Zone trong danh sách trước khi tự chấm điểm!")
            return
        item = self.viewer.zone_items[cur_row]
        num_kpts = self.spn_tab1_pose_kpts.value() if hasattr(self, 'spn_tab1_pose_kpts') else 2
        
        item.keypoints = []
        if hasattr(item, 'polygon'):
            poly = item.polygon()
            count = poly.count()
            if count >= 3:
                pts = np.array([[poly.at(i).x(), poly.at(i).y()] for i in range(count)], dtype=np.float32)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                if num_kpts == 1:
                    item.keypoints.append({'pos': QPointF(cx, cy), 'v': 2})
                else:
                    try:
                        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                        dir_v = eigenvectors[0]
                        proj = np.dot(pts - mean, dir_v.T).squeeze()
                        p_min = float(np.min(proj))
                        p_max = float(np.max(proj))
                        t_vals = np.linspace(p_min * 0.85, p_max * 0.85, num_kpts)
                        for t in t_vals:
                            kp = mean[0] + dir_v * t
                            item.keypoints.append({'pos': QPointF(float(kp[0]), float(kp[1])), 'v': 2})
                    except Exception:
                        r = poly.boundingRect()
                        for i in range(num_kpts):
                            f = (i + 1) / (num_kpts + 1)
                            item.keypoints.append({'pos': QPointF(r.x() + r.width() * f, r.y() + r.height() * f), 'v': 2})
            else:
                return
        else:
            r = item.rect()
            w, h = r.width(), r.height()
            if num_kpts == 1:
                item.keypoints.append({'pos': QPointF(r.x() + w / 2.0, r.y() + h / 2.0), 'v': 2})
            elif num_kpts == 2:
                if w >= h:
                    item.keypoints.append({'pos': QPointF(r.x() + 0.15 * w, r.y() + 0.5 * h), 'v': 2})
                    item.keypoints.append({'pos': QPointF(r.x() + 0.85 * w, r.y() + 0.5 * h), 'v': 2})
                else:
                    item.keypoints.append({'pos': QPointF(r.x() + 0.5 * w, r.y() + 0.15 * h), 'v': 2})
                    item.keypoints.append({'pos': QPointF(r.x() + 0.5 * w, r.y() + 0.85 * h), 'v': 2})
            elif num_kpts == 4:
                item.keypoints.append({'pos': QPointF(r.x(), r.y()), 'v': 2})
                item.keypoints.append({'pos': QPointF(r.x() + w, r.y()), 'v': 2})
                item.keypoints.append({'pos': QPointF(r.x() + w, r.y() + h), 'v': 2})
                item.keypoints.append({'pos': QPointF(r.x(), r.y() + h), 'v': 2})
            else:
                if w >= h:
                    for i in range(num_kpts):
                        f = 0.1 + 0.8 * (i / (num_kpts - 1)) if num_kpts > 1 else 0.5
                        item.keypoints.append({'pos': QPointF(r.x() + f * w, r.y() + 0.5 * h), 'v': 2})
                else:
                    for i in range(num_kpts):
                        f = 0.1 + 0.8 * (i / (num_kpts - 1)) if num_kpts > 1 else 0.5
                        item.keypoints.append({'pos': QPointF(r.x() + 0.5 * w, r.y() + f * h), 'v': 2})

        item.update()
        self.lbl_tab1_kpt_status.setText(f"Zone {cur_row + 1}: Đã tự chấm {len(item.keypoints)}/{num_kpts} điểm")
        self.statusBar().showMessage(f"⚡ Đã tự chấm {len(item.keypoints)} điểm cho Zone {cur_row + 1}", 3000)

    def on_tab1_pose_kpts_changed(self, val):
        self.viewer.num_pose_keypoints = val
        cur_row = self.list_zones.currentRow()
        if 0 <= cur_row < len(self.viewer.zone_items):
            item = self.viewer.zone_items[cur_row]
            kpts_count = len(getattr(item, 'keypoints', []))
            self.lbl_tab1_kpt_status.setText(f"Zone {cur_row + 1}: Đã chấm {kpts_count}/{val} điểm")

    def on_tab1_keypoint_added(self, item, count, total):
        try:
            if item in self.viewer.zone_items:
                idx = self.viewer.zone_items.index(item)
                self.lbl_tab1_kpt_status.setText(f"Zone {idx + 1}: Đã chấm {count}/{total} điểm")
                if count == total:
                    self.statusBar().showMessage(f"✅ Đã chấm đủ {total} điểm cho Zone {idx + 1}!", 3000)
        except (ValueError, Exception):
            pass

    def on_prs_updated_from_viewer(self, tool_name, rect):
        """Khi Search hoặc Pattern bị kéo thả/resize trên viewer Tab 1"""
        idx = 2 if "2" in tool_name else 1
        prs_key = f"prs{idx}"
        if prs_key not in self.recipe_data:
            self.recipe_data[prs_key] = {}

        if "SEARCH" in tool_name:
            self.recipe_data[prs_key]["search_rect"] = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
        elif "PATTERN" in tool_name:
            px, py, pw, ph = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
            self.recipe_data[prs_key]["pattern_rect"] = [px, py, pw, ph]
            self.recipe_data[prs_key]["ref_pt"] = [px, py]
            if self.current_setup_img_gray is not None and getattr(self.current_setup_img_gray, 'size', 0) > 0:
                h, w = self.current_setup_img_gray.shape[:2]
                px_c = max(0, min(px, w - 1))
                py_c = max(0, min(py, h - 1))
                pw_c = max(0, min(pw, w - px_c))
                ph_c = max(0, min(ph, h - py_c))
                if pw_c > 0 and ph_c > 0:
                    pattern_img = self.current_setup_img_gray[py_c:py_c + ph_c, px_c:px_c + pw_c]
                    if pattern_img is not None and pattern_img.size > 0:
                        self.recipe_data[prs_key]["tpl_b64"] = img_to_b64(pattern_img)

        th_spin = getattr(self, f"txt_prs{idx}_th", None)
        if th_spin:
            self.recipe_data[prs_key]["threshold"] = th_spin.value()

        if hasattr(self, 'lbl_prs_status'):
            p1 = bool(self.recipe_data.get('prs1', {}).get('tpl_b64'))
            p2 = bool(self.recipe_data.get('prs2', {}).get('tpl_b64'))
            p1_txt = "✅ Đã nạp" if p1 else "❌ Chưa nạp"
            p2_txt = "✅ Đã nạp" if p2 else "❌ Chưa nạp"
            self.lbl_prs_status.setText(f"Pattern 1: {p1_txt}  |  Pattern 2: {p2_txt}")

    def sync_tab1_to_recipe(self):
        """Đồng bộ toàn bộ Defect Zones, Ignore Zones và PRS từ viewer Tab 1 vào self.recipe_data"""
        all_zones = self._get_zones_data()
        self.recipe_data['defect_zones'] = [z for z in all_zones if not z.get('is_ignore')]
        self.recipe_data['ignore_zones'] = [z for z in all_zones if z.get('is_ignore')]
        self.recipe_data['all_zones'] = all_zones
        if hasattr(self, 'spn_tab1_pose_kpts'):
            self.recipe_data['num_pose_kpts'] = self.spn_tab1_pose_kpts.value()

        for idx in [1, 2]:
            prs_key = f"prs{idx}"
            s_rect = self.viewer.get_item_rect(f"SEARCH{idx}")
            p_rect = self.viewer.get_item_rect(f"PATTERN{idx}")
            if prs_key not in self.recipe_data:
                self.recipe_data[prs_key] = {}
            if s_rect:
                self.recipe_data[prs_key]["search_rect"] = s_rect
            if p_rect:
                self.recipe_data[prs_key]["pattern_rect"] = p_rect
                self.recipe_data[prs_key]["ref_pt"] = [p_rect[0], p_rect[1]]
                if self.current_setup_img_gray is not None and getattr(self.current_setup_img_gray, 'size', 0) > 0:
                    px, py, pw, ph = p_rect
                    h, w = self.current_setup_img_gray.shape[:2]
                    px_c = max(0, min(px, w - 1))
                    py_c = max(0, min(py, h - 1))
                    pw_c = max(0, min(pw, w - px_c))
                    ph_c = max(0, min(ph, h - py_c))
                    if pw_c > 0 and ph_c > 0:
                        tpl = self.current_setup_img_gray[py_c:py_c + ph_c, px_c:px_c + pw_c]
                        if tpl is not None and tpl.size > 0:
                            self.recipe_data[prs_key]["tpl_b64"] = img_to_b64(tpl)
            th_spin = getattr(self, f"txt_prs{idx}_th", None)
            if th_spin:
                self.recipe_data[prs_key]["threshold"] = th_spin.value()

    def set_prs_button_state(self, prs_idx, learned: bool):
        """Cập nhật màu và nhãn nút Learn PRS: Xám khi chưa học, Xanh lá khi đã học thành công"""
        btn = getattr(self, f"btn_learn{prs_idx}", None)
        if btn is None:
            return
        if learned:
            btn.setText(f"✅ Learn PRS {prs_idx} (Đã học)")
            btn.setStyleSheet(
                "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                "stop:0 #2e7d32, stop:1 #43a047); color: white; font-weight: bold;"
                "border: none; border-radius: 6px; padding: 6px; font-size: 12px; }"
                "QPushButton:hover { background: #4caf50; }"
            )
        else:
            btn.setText(f"📐 Learn PRS {prs_idx}")
            btn.setStyleSheet(
                "QPushButton { background: #4a4d52; color: #d0d0d0; font-weight: bold;"
                "border: 1px solid #5a5d62; border-radius: 6px; padding: 6px; font-size: 12px; }"
                "QPushButton:hover { background: #5a5d64; color: white; }"
            )

    def load_setup_image(self):
        """Mở file ảnh master và hiển thị lên viewer, tự động xóa các zone cũ đã vẽ"""
        path, _ = QFileDialog.getOpenFileName(self, "Chọn Ảnh Master", "",
                                               "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)")
        if path:
            if self.viewer.load_image(path):
                self.current_setup_img_gray = cv2_imread(path, cv2.IMREAD_GRAYSCALE)
                self.master_image_path = path
                self.setWindowTitle(f"Defect & Dataset Studio AI — {os.path.basename(path)}")

                # Tự động xóa các zone cũ đã vẽ khi nạp ảnh master mới
                self.clear_all_zones()

                # Đưa 2 nút Learn PRS về màu xám mặc định
                self.set_prs_button_state(1, False)
                self.set_prs_button_state(2, False)
                if hasattr(self, 'recipe_data') and isinstance(self.recipe_data, dict):
                    self.recipe_data["prs1"] = {}
                    self.recipe_data["prs2"] = {}

    def learn_prs(self, prs_idx):
        """Thu thập Pattern từ vùng Search/Pattern đã vẽ"""
        if self.current_setup_img_gray is None:
            QMessageBox.warning(self, "Lỗi", "Hãy nạp Ảnh Master trước khi Learn PRS!")
            return

        search_key = f"SEARCH{prs_idx}"
        pattern_key = f"PATTERN{prs_idx}"
        s_rect = self.viewer.get_item_rect(search_key)
        p_rect = self.viewer.get_item_rect(pattern_key)

        if not s_rect or not p_rect:
            QMessageBox.warning(self, "Thiếu Vùng",
                                 f"Vui lòng vẽ cả Search Area và Pattern cho PRS {prs_idx}")
            return

        px, py, pw, ph = p_rect
        sx, sy, sw, sh = s_rect

        # Kiểm tra Pattern nằm gọn trong Search
        if px < sx or py < sy or (px + pw) > (sx + sw) or (py + ph) > (sy + sh):
            QMessageBox.warning(self, "Cảnh báo", "Vùng Pattern phải nằm gọn bên trong vùng Search!")
            return

        # Cắt ảnh Pattern
        img_h, img_w = self.current_setup_img_gray.shape[:2]
        px = max(0, px); py = max(0, py)
        pw = min(pw, img_w - px); ph = min(ph, img_h - py)
        if pw <= 0 or ph <= 0:
            QMessageBox.warning(self, "Lỗi", "Vùng Pattern không hợp lệ (nằm ngoài ảnh)!")
            return
        pattern_img = self.current_setup_img_gray[py:py + ph, px:px + pw]
        b64_str = img_to_b64(pattern_img)

        # Lấy threshold tương ứng
        if prs_idx == 1:
            threshold = self.txt_prs1_th.value()
        else:
            threshold = self.txt_prs2_th.value()

        # Lưu vào recipe_data
        recipe_key = f"prs{prs_idx}"
        self.recipe_data[recipe_key] = {
            "search_rect": s_rect,
            "pattern_rect": p_rect,
            "ref_pt": [px, py],
            "tpl_b64": b64_str,
            "threshold": threshold
        }

        # Cập nhật nút PRS sang màu xanh lá
        self.set_prs_button_state(prs_idx, True)

        QMessageBox.information(self, "Thành công",
                                 f"Đã Learn PRS {prs_idx} thành công!\n"
                                 f"Pattern size: {pw}x{ph}\n"
                                 f"Threshold: {threshold}")

    # --- ZONE CALLBACKS ---
    def on_zone_added_from_viewer(self, rect):
        """Khi vẽ xong Defect Zone mới trên viewer"""
        zone_idx = self.list_zones.count()
        item = self.viewer.zone_items[-1] if self.viewer.zone_items else None
        is_ignore = item and "IGNORE" in getattr(item, 'item_type', '')
        prefix = "🚫 [Ignore] " if is_ignore else ""
        name = f"{prefix}Zone_{zone_idx + 1}"
        
        if item and hasattr(item, 'polygon'):
            pts = item.polygon().count()
            self.list_zones.addItem(f"{name}: Polygon ({pts} pts)")
        else:
            self.list_zones.addItem(f"{name}: X:{int(rect.x())} Y:{int(rect.y())} "
                                     f"W:{int(rect.width())} H:{int(rect.height())}")
        self.list_zones.setCurrentRow(self.list_zones.count() - 1)

    def on_zone_selected_from_viewer(self, idx, rect):
        """Khi chọn Zone trên viewer"""
        self.ui_is_updating = True
        self.list_zones.blockSignals(True)
        self.list_zones.setCurrentRow(idx)
        self.list_zones.blockSignals(False)
        self.txt_zx.setText(str(int(rect.x())))
        self.txt_zy.setText(str(int(rect.y())))
        self.txt_zw.setText(str(int(rect.width())))
        self.txt_zh.setText(str(int(rect.height())))
        self.ui_is_updating = False

    def on_zone_updated_from_viewer(self, idx, rect):
        """Khi Zone bị kéo/resize trên viewer - cập nhật mượt mà nhẹ nhàng"""
        if idx < self.list_zones.count():
            item_widget = self.list_zones.item(idx)
            text = item_widget.text()
            name = text.split(":")[0].strip() if ":" in text else f"Zone_{idx + 1}"
            
            if idx < len(self.viewer.zone_items) and hasattr(self.viewer.zone_items[idx], 'polygon'):
                pts = self.viewer.zone_items[idx].polygon().count()
                item_widget.setText(f"{name}: Polygon ({pts} pts)")
            else:
                item_widget.setText(
                    f"{name}: X:{int(rect.x())} Y:{int(rect.y())} "
                    f"W:{int(rect.width())} H:{int(rect.height())}")

            if self.list_zones.currentRow() == idx:
                self.ui_is_updating = True
                self.txt_zx.setText(str(int(rect.x())))
                self.txt_zy.setText(str(int(rect.y())))
                self.txt_zw.setText(str(int(rect.width())))
                self.txt_zh.setText(str(int(rect.height())))
                self.ui_is_updating = False

    def duplicate_selected_zone(self):
        """Nhân bản (clone) Defect Zone đang chọn trên Master Image"""
        cur_row = self.list_zones.currentRow()
        if cur_row < 0 or cur_row >= len(self.viewer.zone_items):
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn 1 Defect Zone trong danh sách để nhân bản!")
            return
        
        item = self.viewer.zone_items[cur_row]
        offset = 25  # Dịch chuyển nhẹ để người dùng nhìn thấy bản sao ngay lập tức
        
        if hasattr(item, 'polygon'):
            old_poly = item.polygon()
            pos = item.pos()
            new_poly = QPolygonF()
            for i in range(old_poly.count()):
                pt = old_poly.at(i)
                new_poly.append(QPointF(pt.x() + pos.x() + offset, pt.y() + pos.y() + offset))
            
            item_type = getattr(item, 'item_type', 'DEFECT_ZONE_POLY')
            new_item = ResizablePolygonItem(new_poly, self.viewer, item_type)
            self.viewer.scene.addItem(new_item)
            self.viewer.zone_items.append(new_item)
            self.viewer.zone_added_signal.emit(new_item.boundingRect())
        else:
            r = item.rect()
            p = item.pos()
            abs_x = r.x() + p.x() + offset
            abs_y = r.y() + p.y() + offset
            item_type = getattr(item, 'item_type', 'DEFECT_ZONE')
            new_item = ResizableRectItem(QRectF(abs_x, abs_y, r.width(), r.height()), self.viewer, item_type)
            self.viewer.scene.addItem(new_item)
            self.viewer.zone_items.append(new_item)
            self.viewer.zone_added_signal.emit(new_item.rect())
        
        # Nhân bản keypoints nếu có
        if hasattr(item, 'keypoints') and item.keypoints:
            new_item.keypoints = []
            for kp in item.keypoints:
                pos = kp.get('pos')
                new_item.keypoints.append({'pos': QPointF(pos.x(), pos.y()), 'v': kp.get('v', 2)})
            new_item.update()

        # Chọn item mới vừa nhân bản
        new_idx = len(self.viewer.zone_items) - 1
        self.viewer.scene.clearSelection()
        self.viewer.zone_items[new_idx].setSelected(True)
        self.list_zones.setCurrentRow(new_idx)

    def on_list_zone_selected(self, idx):
        """Khi chọn Zone từ list"""
        if 0 <= idx < len(self.viewer.zone_items):
            self.ui_is_updating = True
            self.viewer.scene.blockSignals(True)
            self.viewer.scene.clearSelection()
            item = self.viewer.zone_items[idx]
            item.setSelected(True)
            self.viewer.scene.blockSignals(False)
            
            if hasattr(item, 'polygon'):
                r = item.sceneBoundingRect()
                self.txt_zx.setText(str(int(r.x())))
                self.txt_zy.setText(str(int(r.y())))
                self.txt_zw.setText(str(int(r.width())))
                self.txt_zh.setText(str(int(r.height())))
            else:
                r = item.rect()
                p = item.pos()
                self.txt_zx.setText(str(int(r.x() + p.x())))
                self.txt_zy.setText(str(int(r.y() + p.y())))
                self.txt_zw.setText(str(int(r.width())))
                self.txt_zh.setText(str(int(r.height())))

            if hasattr(self, 'spn_tab1_pose_kpts') and hasattr(self, 'lbl_tab1_kpt_status'):
                num_kpts = self.spn_tab1_pose_kpts.value()
                kpts_count = len(getattr(item, 'keypoints', [])) if item else 0
                self.lbl_tab1_kpt_status.setText(f"Zone {idx + 1}: Đã chấm {kpts_count}/{num_kpts} điểm")

            self.ui_is_updating = False

    def on_zone_input_changed(self):
        """Khi user thay đổi X/Y/W/H bằng tay"""
        if self.ui_is_updating:
            return
        idx = self.list_zones.currentRow()
        if 0 <= idx < len(self.viewer.zone_items):
            item = self.viewer.zone_items[idx]
            if hasattr(item, 'polygon'):
                return # Bỏ qua update cho polygon
            try:
                x = int(self.txt_zx.text() or 0)
                y = int(self.txt_zy.text() or 0)
                w = int(self.txt_zw.text() or 1)
                h = int(self.txt_zh.text() or 1)
                self.viewer.update_zone_geometry(idx, x, y, w, h)
                zones_data = self._get_zones_data()
                name = zones_data[idx].get('name', f'Zone_{idx + 1}') if idx < len(zones_data) else f'Zone_{idx + 1}'
                self.list_zones.item(idx).setText(
                    f"{name}: X:{x} Y:{y} W:{w} H:{h}")
            except ValueError:
                pass

    def delete_selected_zone(self):
        idx = self.list_zones.currentRow()
        if idx >= 0:
            self.list_zones.blockSignals(True)
            self.list_zones.takeItem(idx)
            self.list_zones.blockSignals(False)
            self.viewer.remove_zone(idx)
            # Clear coordinate fields if no zone left
            if self.list_zones.count() == 0:
                self.txt_zx.clear()
                self.txt_zy.clear()
                self.txt_zw.clear()
                self.txt_zh.clear()

    def clear_all_zones(self):
        self.list_zones.clear()
        self.txt_zx.clear()
        self.txt_zy.clear()
        self.txt_zw.clear()
        self.txt_zh.clear()
        self.viewer.clear_zones_only()

    def _get_zones_data(self):
        """Tạo danh sách Zone data từ viewer items"""
        zones = []
        for i in range(len(self.viewer.zone_items)):
            item = self.viewer.zone_items[i]
            is_ignore = "IGNORE" in getattr(item, 'item_type', '')
            prefix = "🚫 [Ignore] " if is_ignore else ""
            name = f"{prefix}Zone_{i + 1}"
            if i < self.list_zones.count():
                text = self.list_zones.item(i).text()
                if ":" in text:
                    name = text.split(":")[0].strip()
            
            kpts_data = []
            if hasattr(item, 'keypoints') and item.keypoints:
                for kp in item.keypoints:
                    pt_scene = item.mapToScene(kp['pos'])
                    kpts_data.append([round(float(pt_scene.x()), 2), round(float(pt_scene.y()), 2), int(kp.get('v', 2))])

            if hasattr(item, 'polygon'):
                poly = item.polygon()
                pos = item.pos()
                points = [[int(pos.x() + poly.at(j).x()), int(pos.y() + poly.at(j).y())] for j in range(poly.count())]
                zone_dict = {
                    "type": "polygon",
                    "points": points,
                    "name": name,
                    "is_ignore": is_ignore
                }
            else:
                r = item.rect()
                p = item.pos()
                zone_dict = {
                    "type": "rect",
                    "rect": [int(r.x() + p.x()), int(r.y() + p.y()), int(r.width()), int(r.height())],
                    "name": name,
                    "is_ignore": is_ignore
                }
            if kpts_data:
                zone_dict["keypoints"] = kpts_data
            zones.append(zone_dict)
        return zones

    # --- RECIPE SAVE / LOAD / NEW ---
    def new_recipe(self):
        """Reset toàn bộ recipe"""
        reply = QMessageBox.question(self, "New Recipe",
                                      "Bạn có chắc muốn tạo Recipe mới? Dữ liệu chưa lưu sẽ mất.")
        if reply == QMessageBox.StandardButton.Yes:
            self.recipe_data = {"master_image_b64": "", "prs1": {}, "prs2": {}, "defect_zones": []}
            self.current_setup_img_gray = None
            self.master_image_path = ""
            self.viewer.scene.clear()
            self.viewer.zone_items.clear()
            self.viewer.prs_items.clear()
            self.viewer.image_item = None
            self.list_zones.clear()
            self.txt_zx.clear()
            self.txt_zy.clear()
            self.txt_zw.clear()
            self.txt_zh.clear()
            self.txt_prs1_th.setValue(0.7)
            self.txt_prs2_th.setValue(0.7)
            if hasattr(self, 'spn_tab1_pose_kpts'):
                self.spn_tab1_pose_kpts.setValue(2)
            if hasattr(self, 'lbl_tab1_kpt_status'):
                self.lbl_tab1_kpt_status.setText("Trạng thái: Chưa chọn Zone")
            self.set_prs_button_state(1, False)
            self.set_prs_button_state(2, False)
            self.setWindowTitle("Defect & Dataset Studio AI — New Recipe")

    def save_brush_preset(self):
        """Lưu toàn bộ thông số brush vào file JSON"""
        preset = self._get_brush_params()
        preset['brush_types'] = self._get_brush_types()
        preset['defect_mode'] = self.cmb_mode.currentIndex()
        preset['surface_params'] = self._get_surface_params()
        
        path, _ = QFileDialog.getSaveFileName(self, "Lưu Brush Preset", "", "JSON (*.json)")
        if path:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(preset, f, indent=4)
            QMessageBox.information(self, "OK", "Đã lưu Brush Preset thành công!")

    def load_brush_preset(self):
        """Load toàn bộ thông số brush từ file JSON"""
        path, _ = QFileDialog.getOpenFileName(self, "Mở Brush Preset", "", "JSON (*.json)")
        if not path:
            return
            
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
                
            if 'brush_types' in preset:
                for key, chk in self.chk_brushes.items():
                    chk.setChecked(key in preset['brush_types'])
            elif 'brush_type' in preset: # Tương thích ngược với preset cũ
                pass
            if 'defect_mode' in preset:
                self.cmb_mode.setCurrentIndex(preset['defect_mode'])
                
            if 'size_min' in preset: self.spn_size_min.setValue(preset['size_min'])
            if 'size_max' in preset: self.spn_size_max.setValue(preset['size_max'])
            if 'opacity_min' in preset: self.spn_opacity_min.setValue(preset['opacity_min'])
            if 'opacity_max' in preset: self.spn_opacity_max.setValue(preset['opacity_max'])
            if 'intensity_min' in preset: self.spn_int_min.setValue(preset['intensity_min'])
            if 'intensity_max' in preset: self.spn_int_max.setValue(preset['intensity_max'])
            if 'count_min' in preset: self.spn_count_min.setValue(preset['count_min'])
            if 'count_max' in preset: self.spn_count_max.setValue(preset['count_max'])
            if 'length_min' in preset: self.spn_length_min.setValue(preset['length_min'])
            if 'length_max' in preset: self.spn_length_max.setValue(preset['length_max'])
            if 'angle_min' in preset: self.spn_angle_min.setValue(preset['angle_min'])
            if 'angle_max' in preset: self.spn_angle_max.setValue(preset['angle_max'])
            if 'main_thickness_min' in preset: self.spn_main_thick_min.setValue(preset['main_thickness_min'])
            elif 'main_thick_min' in preset: self.spn_main_thick_min.setValue(preset['main_thick_min'])
            if 'main_thickness_max' in preset: self.spn_main_thick_max.setValue(preset['main_thickness_max'])
            elif 'main_thick_max' in preset: self.spn_main_thick_max.setValue(preset['main_thick_max'])
            if 'sub_branch_min' in preset: self.spn_sub_min.setValue(preset['sub_branch_min'])
            if 'sub_branch_max' in preset: self.spn_sub_max.setValue(preset['sub_branch_max'])
            if 'sub_branch_scale' in preset: self.spn_sub_scale.setValue(preset['sub_branch_scale'])
            if 'stamp_mode' in preset and hasattr(self, 'cmb_stamp_mode'):
                sm = str(preset['stamp_mode'])
                for s_idx in range(self.cmb_stamp_mode.count()):
                    if sm in self.cmb_stamp_mode.itemText(s_idx):
                        self.cmb_stamp_mode.setCurrentIndex(s_idx)
                        break
            if 'count_scope' in preset and hasattr(self, 'cmb_count_scope'):
                for s_idx in range(self.cmb_count_scope.count()):
                    if str(preset['count_scope']) in self.cmb_count_scope.itemText(s_idx):
                        self.cmb_count_scope.setCurrentIndex(s_idx)
                        break
            if 'surface_params' in preset:
                sp = preset['surface_params']
                if 'enabled' in sp: self.chk_auto_surface.setChecked(sp['enabled'])
                if 'target_surface' in sp:
                    ts = sp['target_surface']
                    for idx in range(self.cmb_target_surface.count()):
                        if ts in self.cmb_target_surface.itemText(idx):
                            self.cmb_target_surface.setCurrentIndex(idx)
                            break
                if 'threshold_mode' in sp:
                    tm = sp['threshold_mode']
                    for idx in range(self.cmb_thresh_mode.count()):
                        if tm in self.cmb_thresh_mode.itemText(idx):
                            self.cmb_thresh_mode.setCurrentIndex(idx)
                            break
                if 'manual_threshold' in sp: self.spn_manual_thresh.setValue(sp['manual_threshold'])
                if 'margin' in sp: self.spn_surface_margin.setValue(sp['margin'])
                if 'min_area' in sp: self.spn_surface_min_area.setValue(sp['min_area'])
            
            QMessageBox.information(self, "OK", "Đã nạp Brush Preset thành công!")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể load preset: {e}")

    def save_recipe(self):
        """Lưu Recipe ra file JSON (bao gồm master image base64)"""
        # Đồng bộ toàn bộ Defect Zones, Ignore Zones và PRS từ viewer Tab 1
        self.sync_tab1_to_recipe()

        # Lưu master image
        if self.current_setup_img_gray is not None:
            self.recipe_data['master_image_b64'] = img_to_b64(self.current_setup_img_gray)

        path, _ = QFileDialog.getSaveFileName(self, "Lưu Recipe", "", "JSON (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.recipe_data, f, indent=4)
            QMessageBox.information(self, "OK", f"Đã lưu Recipe thành công!\n{path}")

    def load_recipe(self):
        """Load Recipe từ file JSON, khôi phục master image + PRS + Zones"""
        path, _ = QFileDialog.getOpenFileName(self, "Mở Recipe", "", "JSON (*.json)")
        if not path:
            return

        with open(path, 'r', encoding='utf-8') as f:
            self.recipe_data = json.load(f)

        # Khôi phục master image
        master_b64 = self.recipe_data.get('master_image_b64', '')
        if master_b64:
            self.current_setup_img_gray = b64_to_img(master_b64)
            if self.current_setup_img_gray is not None:
                self.viewer.load_image_from_array(self.current_setup_img_gray)
        else:
            self.current_setup_img_gray = None
            self.viewer.scene.clear()
            self.viewer.zone_items.clear()
            self.viewer.prs_items.clear()

        # Xóa các zone cũ trên viewer trước khi khôi phục
        self.viewer.clear_all()

        prs1 = self.recipe_data.get('prs1', {})
        prs2 = self.recipe_data.get('prs2', {})

        if prs1:
            if 'search_rect' in prs1:
                self.viewer.draw_prs_from_dict("SEARCH1", prs1['search_rect'])
            if 'pattern_rect' in prs1:
                self.viewer.draw_prs_from_dict("PATTERN1", prs1['pattern_rect'])
            self.txt_prs1_th.setValue(prs1.get('threshold', 0.7))

        if prs2:
            if 'search_rect' in prs2:
                self.viewer.draw_prs_from_dict("SEARCH2", prs2['search_rect'])
            if 'pattern_rect' in prs2:
                self.viewer.draw_prs_from_dict("PATTERN2", prs2['pattern_rect'])
            self.txt_prs2_th.setValue(prs2.get('threshold', 0.7))

        self.set_prs_button_state(1, bool(prs1 and prs1.get('tpl_b64')))
        self.set_prs_button_state(2, bool(prs2 and prs2.get('tpl_b64')))

        # Khôi phục Defect Zones & Ignore Zones
        self.list_zones.clear()
        zones = self.recipe_data.get('all_zones')
        if zones is None:
            zones = self.recipe_data.get('defect_zones', []) + self.recipe_data.get('ignore_zones', [])
        self.viewer.draw_zones_from_list(zones)
        for i, zone in enumerate(zones):
            name = zone.get('name', f'Zone_{i + 1}') if isinstance(zone, dict) else f'Zone_{i + 1}'
            if isinstance(zone, dict) and zone.get("type") == "polygon":
                pts = len(zone.get("points", []))
                self.list_zones.addItem(f"{name}: Polygon ({pts} pts)")
            else:
                rect = zone['rect'] if isinstance(zone, dict) else zone
                x, y, w, h = rect
                self.list_zones.addItem(f"{name}: X:{x} Y:{y} W:{w} H:{h}")

        if 'num_pose_kpts' in self.recipe_data and hasattr(self, 'spn_tab1_pose_kpts'):
            self.spn_tab1_pose_kpts.setValue(int(self.recipe_data['num_pose_kpts']))

        self.setWindowTitle(f"Defect & Dataset Studio AI — {os.path.basename(path)}")

    # ==========================================
    # TAB 2: BATCH GENERATE LOGIC
    # ==========================================
    def browse_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn Folder ảnh Input")
        if path:
            self.txt_input_folder.setText(path)
            files = self._get_image_files(path)
            self.lbl_file_count.setText(f"{len(files)} ảnh")

    def browse_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn Folder Output")
        if path:
            self.txt_output_folder.setText(path)

    def browse_labels_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn Folder Labels (YOLO)")
        if path:
            self.txt_labels_folder.setText(path)

    def _get_brush_types(self):
        """Trả về danh sách các brush type được chọn"""
        selected = [key for key, chk in self.chk_brushes.items() if chk.isChecked()]
        if not selected or 'random' in selected:
            return ['random']
        return selected

    def _get_brush_params(self):
        """Thu thập brush params từ UI, đảm bảo min luôn <= max"""
        return {
            'size_min': min(self.spn_size_min.value(), self.spn_size_max.value()),
            'size_max': max(self.spn_size_min.value(), self.spn_size_max.value()),
            'opacity_min': min(self.spn_opacity_min.value(), self.spn_opacity_max.value()),
            'opacity_max': max(self.spn_opacity_min.value(), self.spn_opacity_max.value()),
            'intensity_min': min(self.spn_int_min.value(), self.spn_int_max.value()),
            'intensity_max': max(self.spn_int_min.value(), self.spn_int_max.value()),
            'count_min': min(self.spn_count_min.value(), self.spn_count_max.value()),
            'count_max': max(self.spn_count_min.value(), self.spn_count_max.value()),
            # Tham số mở rộng
            'length_min': min(self.spn_length_min.value(), self.spn_length_max.value()),
            'length_max': max(self.spn_length_min.value(), self.spn_length_max.value()),
            'angle_min': min(self.spn_angle_min.value(), self.spn_angle_max.value()),
            'angle_max': max(self.spn_angle_min.value(), self.spn_angle_max.value()),
            'main_thickness_min': min(self.spn_main_thick_min.value(), self.spn_main_thick_max.value()),
            'main_thickness_max': max(self.spn_main_thick_min.value(), self.spn_main_thick_max.value()),
            'sub_branch_min': min(self.spn_sub_min.value(), self.spn_sub_max.value()),
            'sub_branch_max': max(self.spn_sub_min.value(), self.spn_sub_max.value()),
            'sub_branch_scale': self.spn_sub_scale.value(),
            'stamp_mode': self.cmb_stamp_mode.currentText().split(' — ')[0].strip() if hasattr(self, 'cmb_stamp_mode') else 'mixed',
            'count_scope': self.cmb_count_scope.currentText().split(' — ')[0].strip() if hasattr(self, 'cmb_count_scope') else 'per_image',
        }

    def _get_defect_mode(self):
        """Lấy defect mode từ combo: 0=multiply (tự nhiên), 1=darken, 2=lighten"""
        idx = self.cmb_mode.currentIndex()
        if idx == 0:
            return "multiply"
        elif idx == 1:
            return "darken"
        elif idx == 2:
            return "lighten"
        return "multiply"

    def _get_surface_params(self):
        """Lấy cấu hình Auto Surface Detection"""
        return {
            'enabled': self.chk_auto_surface.isChecked(),
            'target_surface': self.cmb_target_surface.currentText().split(' — ')[0].strip(),
            'threshold_mode': self.cmb_thresh_mode.currentText().split(' — ')[0].strip(),
            'manual_threshold': self.spn_manual_thresh.value(),
            'margin': self.spn_surface_margin.value(),
            'min_area': self.spn_surface_min_area.value(),
        }

    def choose_custom_stamp_image(self):
        """Cho phép người dùng chọn ảnh mẫu texture khác để nạp vào StampBank"""
        from brush_engine import StampBank
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Ảnh Mẫu Lỗi (Texture/Stamps)", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            count = StampBank.load_custom_image(path)
            fname = os.path.basename(path)
            if hasattr(self, 'lbl_stamp_count'):
                self.lbl_stamp_count.setText(f"Đã nạp: {count} mẫu từ {fname}")
            QMessageBox.information(
                self, "Thành công",
                f"Đã trích xuất thành công {count} mẫu lỗi tự nhiên từ:\n{fname}\n\n"
                f"Hệ thống sẽ sử dụng các mẫu này khi chọn loại brush '✨ Mẫu thực (Stamp)'."
            )

    def reset_default_stamps(self):
        """Khôi phục về các mẫu lỗi thực tế mặc định (từ thư viện chuẩn)"""
        from brush_engine import StampBank
        StampBank.reset_to_default()
        count = StampBank.count()
        if hasattr(self, 'lbl_stamp_count'):
            self.lbl_stamp_count.setText(f"Đã nạp: {count}+ mẫu thực tế")
        QMessageBox.information(self, "Thông báo", f"Đã khôi phục {count} mẫu lỗi tự nhiên mặc định!")

    def preview_single(self):
        """Preview brush trên 1 ảnh đầu tiên trong folder (hoặc master image)"""
        try:
            # Tìm ảnh để test
            test_img = None
            input_folder = self.txt_input_folder.text()
            if input_folder and os.path.isdir(input_folder):
                files = self._get_image_files(input_folder)
                if files:
                    test_img = cv2_imread(files[0], cv2.IMREAD_GRAYSCALE)

            if test_img is None and self.current_setup_img_gray is not None:
                test_img = self.current_setup_img_gray.copy()

            if test_img is None:
                QMessageBox.warning(self, "Lỗi",
                                     "Hãy nạp ảnh master hoặc chọn folder input trước!")
                return

            brush_types = self._get_brush_types()
            brush_type = random.choice(brush_types) if brush_types else 'random'
            if brush_type == 'random':
                brush_type = random.choice(BrushEngine.ALL_TYPES)
            brush_params = self._get_brush_params()
            defect_mode = self._get_defect_mode()
            img_h, img_w = test_img.shape[:2]
            result = test_img.copy()

            surf_params = self._get_surface_params()
            if surf_params.get('enabled', False):
                # CHẾ ĐỘ AUTO SURFACE DETECTION: Tự động phân đoạn vùng sáng/tối
                zones = BrushEngine.detect_surfaces(
                    result,
                    target_surface=surf_params.get('target_surface', 'bright'),
                    threshold_mode=surf_params.get('threshold_mode', 'otsu'),
                    manual_threshold=surf_params.get('manual_threshold', 128),
                    margin=surf_params.get('margin', 4),
                    min_area=surf_params.get('min_area', 200)
                )
                if not zones:
                    QMessageBox.warning(self, "Thông báo",
                                        "Không tìm thấy vùng bề mặt phù hợp với ngưỡng hiện tại!\n"
                                        "Hãy kiểm tra lại thông số diện tích min hoặc chế độ phân ngưỡng.")
                    return

                applied_count = 0
                zone_counts = BrushEngine.allocate_counts([z['area'] for z in zones], brush_params)
                for z_idx, zone in enumerate(zones):
                    if zone_counts[z_idx] == 0:
                        continue
                    zone_params = dict(brush_params, count_min=zone_counts[z_idx], count_max=zone_counts[z_idx])
                    zx, zy, zw, zh = zone['rect']
                    if zw < 5 or zh < 5:
                        continue

                    # Sinh lại tối đa 15 lần cho tới khi vết lỗi nằm trên bề mặt (đã qua Margin Erosion)
                    surface_mask = zone['surface_mask']
                    best = None
                    for _try in range(15):
                        raw_mask, bboxes, actual_type = BrushEngine.generate_defect(zw, zh, brush_type, zone_params)
                        raw_px = np.count_nonzero(raw_mask)
                        clipped = cv2.bitwise_and(raw_mask, surface_mask)
                        ratio = np.count_nonzero(clipped) / raw_px if raw_px else 0.0
                        if best is None or ratio > best[0]:
                            best = (ratio, clipped, bboxes, actual_type)
                        if ratio >= 0.95:  # vết nằm trọn trong bề mặt -> không bị cắt vụn
                            break
                    _, mask, bboxes, actual_type = best
                    if np.count_nonzero(mask) == 0:
                        continue

                    opacity = random.uniform(brush_params['opacity_min'], brush_params['opacity_max'])
                    current_mode = zone['defect_mode'] if surf_params.get('target_surface') == 'both' else defect_mode
                    result = BrushEngine.apply_defect(result, [zx, zy, zw, zh], mask, opacity, current_mode)
                    applied_count += 1

                self.batch_viewer.display_result_image(result)
                self._append_log(self.batch_log, f"👁 Preview Auto Surface: Đã sinh lỗi trên {applied_count}/{len(zones)} vùng bề mặt ({brush_type})")
                return

            # CHẾ ĐỘ THƯỜNG DỰA TRÊN RECIPE DEFECT ZONES
            all_zones = self._get_zones_data()
            self.recipe_data['defect_zones'] = [z for z in all_zones if not z.get('is_ignore')]
            self.recipe_data['ignore_zones'] = [z for z in all_zones if z.get('is_ignore')]
            zones = self.recipe_data.get('defect_zones', [])
            if not zones:
                QMessageBox.warning(self, "Lỗi", "Hãy vẽ ít nhất 1 Defect Zone trên Tab 1 hoặc bật 'Tự động dò bề mặt'!")
                return

            zone_counts = BrushEngine.allocate_counts([(cv2.contourArea(np.array(z['points'], dtype=np.int32)) if z.get('type') == 'polygon' else float(z['rect'][2]) * float(z['rect'][3])) for z in zones], brush_params)
            for z_idx, zone in enumerate(zones):
                if zone_counts[z_idx] == 0:
                    continue
                zone_params = dict(brush_params, count_min=zone_counts[z_idx], count_max=zone_counts[z_idx])
                if zone.get("type") == "polygon":
                    pts = np.array(zone["points"], dtype=np.int32)
                    zx, zy, zw, zh = cv2.boundingRect(pts)
                else:
                    rect = zone['rect'] if isinstance(zone, dict) else zone
                    zx, zy, zw, zh = map(int, rect)

                zx = max(0, min(zx, img_w - 1))
                zy = max(0, min(zy, img_h - 1))
                zw = max(1, min(zw, img_w - zx))
                zh = max(1, min(zh, img_h - zy))
                if zw < 5 or zh < 5:
                    continue

                mask, bboxes, actual_type = BrushEngine.generate_defect(zw, zh, brush_type, zone_params)

                # Nếu là polygon, cắt mask theo hình đa giác
                if zone.get("type") == "polygon":
                    local_pts = pts - np.array([zx, zy])
                    poly_mask = np.zeros((zh, zw), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [local_pts], 255)
                    mask = cv2.bitwise_and(mask, poly_mask)

                opacity = random.uniform(brush_params['opacity_min'], brush_params['opacity_max'])
                result = BrushEngine.apply_defect(result, [zx, zy, zw, zh], mask, opacity, defect_mode)

            # Hiển thị
            self.batch_viewer.display_result_image(result)
            self._append_log(self.batch_log, f"👁 Preview: Đã tạo defect ({brush_type}) trên các Defect Zone")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi Preview", f"Đã xảy ra lỗi khi tạo preview:\n{e}")
            if hasattr(self, 'batch_log'):
                self._append_log(self.batch_log, f"❌ Lỗi Preview: {e}")

    def toggle_run(self):
        """Bắt đầu / Dừng batch processing"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.btn_run.setText("▶ RUN ALL — Tạo Defect Hàng Loạt")
            self.btn_run.setStyleSheet("""
                QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                              stop:0 #4CAF50, stop:1 #66BB6A);
                              color: white; font-weight: bold; font-size: 15px;
                              border: none; border-radius: 10px; }
                QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                    stop:0 #66BB6A, stop:1 #81C784); }
            """)
            return

        # Validate
        input_folder = self.txt_input_folder.text()
        output_folder = self.txt_output_folder.text()

        if not input_folder or not os.path.isdir(input_folder):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn folder input hợp lệ!")
            return

        if not output_folder:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn folder output!")
            return

        files = self._get_image_files(input_folder)
        if not files:
            QMessageBox.warning(self, "Lỗi", "Folder input không chứa file ảnh!")
            return

        surf_params = self._get_surface_params()
        if not surf_params.get('enabled', False):
            # Cập nhật zones data nếu không dùng auto surface
            all_zones = self._get_zones_data()
            self.recipe_data['defect_zones'] = [z for z in all_zones if not z.get('is_ignore')]
            self.recipe_data['ignore_zones'] = [z for z in all_zones if z.get('is_ignore')]
            if not self.recipe_data.get('defect_zones'):
                QMessageBox.warning(self, "Lỗi", "Hãy vẽ ít nhất 1 Defect Zone trên Tab 1 hoặc bật 'Tự động dò bề mặt'!")
                return

        # Cập nhật PRS thresholds
        if self.recipe_data.get('prs1'):
            self.recipe_data['prs1']['threshold'] = self.txt_prs1_th.value()
        if self.recipe_data.get('prs2'):
            self.recipe_data['prs2']['threshold'] = self.txt_prs2_th.value()

        # Build config
        config = {
            'recipe': self.recipe_data,
            'input_files': files,
            'output_folder': output_folder,
            'output_format': self.cmb_format.currentText(),
            'brush_types': self._get_brush_types(),
            'brush_params': self._get_brush_params(),
            'defect_mode': self._get_defect_mode(),
            'surface_params': surf_params,
            'yolo_export': self.chk_yolo.isChecked(),
            'yolo_format': self.cmb_yolo_format.currentText(),
            'labels_folder': self.txt_labels_folder.text(),
        }

        # Tạo worker
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.worker = DefectWorkerThread(config)
        self.worker.log_signal.connect(lambda msg: self._append_log(self.batch_log, msg))
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.preview_signal.connect(self._on_preview)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

        # Update UI
        self.btn_run.setText("⏹ DỪNG — Stop")
        self.btn_run.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #F44336, stop:1 #E53935);
                          color: white; font-weight: bold; font-size: 15px;
                          border: none; border-radius: 10px; }
            QPushButton:hover { background: #EF5350; }
        """)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)

    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)

    def _on_preview(self, img, filepath):
        self.batch_viewer.display_result_image(img)

    def _on_finished(self, success_count, total_count):
        self.btn_run.setText("▶ RUN ALL — Tạo Defect Hàng Loạt")
        self.btn_run.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #4CAF50, stop:1 #66BB6A);
                          color: white; font-weight: bold; font-size: 15px;
                          border: none; border-radius: 10px; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                stop:0 #66BB6A, stop:1 #81C784); }
        """)
        QMessageBox.information(self, "Hoàn tất",
                                 f"Đã xử lý xong!\n"
                                 f"Thành công: {success_count}/{total_count} ảnh")

    # ==========================================
    # TIỆN ÍCH
    # ==========================================
    def _append_log(self, log_widget, msg):
        """Thêm dòng log vào TextEdit và auto-scroll"""
        log_widget.append(msg)
        cursor = log_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        log_widget.setTextCursor(cursor)

    # ==========================================
    # TAB 3: YOLO LABEL VIEWER & EDITOR
    # ==========================================
    def setup_tab_viewer(self):
        layout = QHBoxLayout()
        
        left_panel = QWidget()
        left_panel.setMinimumWidth(280)
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(4,4,4,4)
        
        btn_browse = QPushButton("📂 Chọn thư mục (Images + Labels)")
        btn_browse.clicked.connect(self.browse_yolo_folder)
        left_lay.addWidget(btn_browse)
        
        self.list_yolo_images = QListWidget()
        self.list_yolo_images.currentItemChanged.connect(self.on_yolo_image_changed)
        left_lay.addWidget(self.list_yolo_images)

        btn_del_file = QPushButton("🗑️ Xóa file ảnh & nhãn (Del)")
        btn_del_file.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        btn_del_file.setToolTip("Xóa vĩnh viễn file ảnh và nhãn đang xem khỏi đĩa (Phím tắt: Del khi không chọn nhãn)")
        btn_del_file.clicked.connect(self.delete_current_yolo_file)
        left_lay.addWidget(btn_del_file)

        # --- Thêm Toolbar vẽ tay ---
        grp_draw = QGroupBox("✍ Vẽ Manual (YOLO)")
        lay_draw = QVBoxLayout()
        
        # Class Selection & Management
        lay_class_header = QHBoxLayout()
        lbl_class = QLabel("🏷 Chọn Class:")
        lbl_class.setStyleSheet("font-weight: bold; color: #E0E0E0;")
        lay_class_header.addWidget(lbl_class)
        lay_draw.addLayout(lay_class_header)
        
        self.cmb_yolo_class = QComboBox()
        self.cmb_yolo_class.setMinimumHeight(32)
        self.cmb_yolo_class.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.cmb_yolo_class.currentIndexChanged.connect(self.on_yolo_class_changed)
        lay_draw.addWidget(self.cmb_yolo_class)
        
        lay_class_btns = QHBoxLayout()
        btn_add_class = QPushButton("➕ Thêm")
        btn_add_class.setMinimumHeight(28)
        btn_add_class.clicked.connect(self.add_yolo_class)
        
        btn_del_class = QPushButton("❌ Xóa")
        btn_del_class.setMinimumHeight(28)
        btn_del_class.clicked.connect(self.del_yolo_class)
        
        btn_clear_classes = QPushButton("🧹 Xóa hết")
        btn_clear_classes.setMinimumHeight(28)
        btn_clear_classes.setToolTip("Xóa tất cả các class để tạo lại từ đầu")
        btn_clear_classes.clicked.connect(self.clear_yolo_classes)
        
        lay_class_btns.addWidget(btn_add_class)
        lay_class_btns.addWidget(btn_del_class)
        lay_class_btns.addWidget(btn_clear_classes)
        lay_draw.addLayout(lay_class_btns)

        self.yolo_tool_group = QButtonGroup(self)
        btn_none = QRadioButton("🖱 Chọn / Di chuyển")
        btn_none.setChecked(True)
        btn_bbox = QRadioButton("🟩 Vẽ Bounding Box")
        btn_poly = QRadioButton("🔺 Vẽ Đa giác (Thủ công)")
        btn_auto_seg = QRadioButton("🪄 Auto Segment (Box ➔ Đa giác)")
        
        btn_pose = QRadioButton("📍 Vẽ Pose (Keypoints)")
        self.spn_pose_kpts = QSpinBox()
        self.spn_pose_kpts.setPrefix("Số Kpts: ")
        self.spn_pose_kpts.setRange(1, 99)
        self.spn_pose_kpts.setValue(2)
        self.spn_pose_kpts.valueChanged.connect(self.on_pose_kpts_changed)
        
        self.yolo_tool_group.addButton(btn_none, 0)
        self.yolo_tool_group.addButton(btn_bbox, 1)
        self.yolo_tool_group.addButton(btn_poly, 2)
        self.yolo_tool_group.addButton(btn_pose, 3)
        self.yolo_tool_group.addButton(btn_auto_seg, 4)
        
        self.yolo_tool_group.idClicked.connect(self.on_yolo_tool_changed)
        
        lay_draw.addWidget(btn_none)
        lay_draw.addWidget(btn_bbox)
        lay_draw.addWidget(btn_poly)
        
        # Auto Segment Block với tùy chỉnh Threshold & Slider
        lay_auto_seg = QVBoxLayout()
        lay_auto_seg.addWidget(btn_auto_seg)
        
        lay_auto_seg_opts = QHBoxLayout()
        self.chk_auto_seg_inv = QCheckBox("Lỗi tối")
        self.chk_auto_seg_inv.setChecked(True)
        self.chk_auto_seg_inv.setToolTip("Tích chọn nếu lỗi tối hơn nền; Bỏ chọn nếu lỗi sáng hơn nền")
        self.chk_auto_seg_inv.toggled.connect(self.on_auto_seg_opts_changed)
        
        self.chk_auto_seg_otsu = QCheckBox("Tự động (Otsu)")
        self.chk_auto_seg_otsu.setChecked(False)
        self.chk_auto_seg_otsu.toggled.connect(self.on_auto_seg_otsu_toggled)
        
        lay_auto_seg_opts.addWidget(self.chk_auto_seg_inv)
        lay_auto_seg_opts.addWidget(self.chk_auto_seg_otsu)
        lay_auto_seg.addLayout(lay_auto_seg_opts)
        
        lay_th_row = QHBoxLayout()
        lay_th_row.addWidget(QLabel("Ngưỡng:"))
        
        from PyQt6.QtWidgets import QSlider
        self.slider_auto_seg_th = QSlider(Qt.Orientation.Horizontal)
        self.slider_auto_seg_th.setRange(0, 255)
        self.slider_auto_seg_th.setValue(128)
        
        self.spn_auto_seg_th = QSpinBox()
        self.spn_auto_seg_th.setRange(0, 255)
        self.spn_auto_seg_th.setValue(128)
        self.spn_auto_seg_th.setFixedWidth(55)
        
        self.slider_auto_seg_th.valueChanged.connect(self.spn_auto_seg_th.setValue)
        self.spn_auto_seg_th.valueChanged.connect(self.slider_auto_seg_th.setValue)
        self.spn_auto_seg_th.valueChanged.connect(self.on_auto_seg_opts_changed)
        
        lay_th_row.addWidget(self.slider_auto_seg_th)
        lay_th_row.addWidget(self.spn_auto_seg_th)
        lay_auto_seg.addLayout(lay_th_row)
        lay_draw.addLayout(lay_auto_seg)
        
        lay_pose = QHBoxLayout()
        lay_pose.addWidget(btn_pose)
        lay_pose.addWidget(self.spn_pose_kpts)
        lay_draw.addLayout(lay_pose)
        
        lay_del = QHBoxLayout()
        btn_del = QPushButton("✂ Xóa Nhãn Chọn")
        btn_del.setToolTip("Xóa annotation (bbox/polygon/pose) đang được chọn trong ảnh")
        btn_del.clicked.connect(self.delete_selected_yolo)
        btn_clear = QPushButton("🗑 Xóa Hết Nhãn")
        btn_clear.setToolTip("Xóa tất cả các annotation đang có trên ảnh hiện tại")
        btn_clear.clicked.connect(self.clear_all_yolo)
        lay_del.addWidget(btn_del)
        lay_del.addWidget(btn_clear)
        lay_draw.addLayout(lay_del)
        
        grp_draw.setLayout(lay_draw)
        left_lay.addWidget(grp_draw)
        
        btn_save = QPushButton("💾 Save Labels (Ghi đè file .txt)")
        btn_save.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4CAF50, stop:1 #66BB6A); color: white; font-weight: bold; border-radius: 6px; padding: 10px;")
        btn_save.clicked.connect(self.save_yolo_labels)
        left_lay.addWidget(btn_save)
        
        self.yolo_viewer = ImageViewer()
        self.yolo_viewer.current_tool = "NONE"
        self.yolo_viewer.drawing_poly = False
        self.yolo_viewer.on_delete_unselected = self.delete_current_yolo_file

        orig_list_key_press = self.list_yolo_images.keyPressEvent
        def list_yolo_key_press(event):
            if event.key() == Qt.Key.Key_Delete:
                self.delete_current_yolo_file()
                event.accept()
                return
            orig_list_key_press(event)
        self.list_yolo_images.keyPressEvent = list_yolo_key_press
        
        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(self.yolo_viewer, stretch=4)
        self.tab_viewer.setLayout(layout)
        
        self.current_yolo_folder = ""
        self.current_yolo_img_path = ""
        self.current_yolo_txt_path = ""

    def _update_yolo_class_names(self):
        d = {}
        import re
        for i in range(self.cmb_yolo_class.count()):
            text = self.cmb_yolo_class.itemText(i)
            m = re.match(r"^(\d+)\s*[-:]\s*(.+)$", text)
            if m:
                c_id = int(m.group(1))
            else:
                c_id = i
            d[c_id] = text
        if hasattr(self, 'yolo_viewer'):
            self.yolo_viewer.yolo_class_names = d
            self.yolo_viewer.scene.update()
        self._sync_classes_to_prs()

    def _sync_classes_to_prs(self):
        if hasattr(self, 'cmb_prs_class'):
            curr_idx = self.cmb_prs_class.currentIndex()
            self.cmb_prs_class.blockSignals(True)
            self.cmb_prs_class.clear()
            for i in range(self.cmb_yolo_class.count()):
                self.cmb_prs_class.addItem(self.cmb_yolo_class.itemText(i))
            if curr_idx >= 0 and curr_idx < self.cmb_prs_class.count():
                self.cmb_prs_class.setCurrentIndex(curr_idx)
            elif self.cmb_prs_class.count() > 0:
                self.cmb_prs_class.setCurrentIndex(0)
            self.cmb_prs_class.blockSignals(False)

    def add_prs_class(self):
        self.add_yolo_class()

    def del_prs_class(self):
        idx = self.cmb_prs_class.currentIndex()
        if idx >= 0 and idx < self.cmb_yolo_class.count():
            self.cmb_yolo_class.setCurrentIndex(idx)
            self.del_yolo_class()

    def clear_prs_classes(self):
        self.clear_yolo_classes()

    def add_yolo_class(self):
        from PyQt6.QtWidgets import QInputDialog
        next_id = self.cmb_yolo_class.count()
        new_class, ok = QInputDialog.getText(
            self, "Thêm Class YOLO", 
            f"Nhập tên Class mới (VD: scratch, nứt, móp...):\n(Sẽ tự động gán Class ID = {next_id})"
        )
        if ok and new_class.strip():
            raw = new_class.strip()
            import re
            m = re.match(r"^(\d+)\s*[-:]\s*(.+)$", raw)
            if m:
                text = f"{m.group(1)} - {m.group(2).strip()}"
            else:
                text = f"{next_id} - {raw}"
            self.cmb_yolo_class.addItem(text)
            self.cmb_yolo_class.setCurrentIndex(self.cmb_yolo_class.count() - 1)
            self._update_yolo_class_names()
            self.save_settings()

    def del_yolo_class(self):
        idx = self.cmb_yolo_class.currentIndex()
        if idx >= 0:
            self.cmb_yolo_class.removeItem(idx)
            self._update_yolo_class_names()
            self.save_settings()

    def clear_yolo_classes(self):
        if self.cmb_yolo_class.count() == 0:
            return
        from PyQt6.QtWidgets import QMessageBox
        res = QMessageBox.question(
            self, "Xác nhận xóa hết",
            "Bạn có chắc muốn xóa TẤT CẢ các Class hiện có trong danh sách?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if res == QMessageBox.StandardButton.Yes:
            self.cmb_yolo_class.clear()
            self._update_yolo_class_names()
            self.save_settings()

    def on_yolo_class_changed(self, idx):
        text = self.cmb_yolo_class.currentText()
        if text:
            import re
            m = re.match(r"^(\d+)\s*[-:]\s*(.+)$", text)
            if m:
                c_id = int(m.group(1))
            else:
                c_id = idx if idx >= 0 else 0
            self.yolo_viewer.current_yolo_class_id = c_id
        else:
            self.yolo_viewer.current_yolo_class_id = 0
        self._update_yolo_class_names()

    def on_pose_kpts_changed(self, val):
        if hasattr(self, 'yolo_viewer'):
            self.yolo_viewer.num_pose_keypoints = val

    def on_auto_seg_otsu_toggled(self, checked):
        self.slider_auto_seg_th.setEnabled(not checked)
        self.spn_auto_seg_th.setEnabled(not checked)
        self.on_auto_seg_opts_changed()

    def on_auto_seg_opts_changed(self):
        if hasattr(self, 'yolo_viewer'):
            self.yolo_viewer.auto_seg_otsu = self.chk_auto_seg_otsu.isChecked()
            self.yolo_viewer.auto_seg_invert = self.chk_auto_seg_inv.isChecked()
            self.yolo_viewer.auto_seg_th = self.spn_auto_seg_th.value()
            if hasattr(self.yolo_viewer, 'reapply_last_auto_seg'):
                self.yolo_viewer.reapply_last_auto_seg()

    def on_yolo_tool_changed(self, idx):
        if idx == 0:
            self.yolo_viewer.current_tool = "NONE"
        elif idx == 1:
            self.yolo_viewer.current_tool = "YOLO_BBOX"
        elif idx == 2:
            self.yolo_viewer.current_tool = "YOLO_POLY"
        elif idx == 3:
            self.yolo_viewer.current_tool = "YOLO_POSE"
            self.yolo_viewer.num_pose_keypoints = self.spn_pose_kpts.value()
        elif idx == 4:
            self.yolo_viewer.current_tool = "YOLO_AUTO_SEG"
            self.on_auto_seg_opts_changed()

    def delete_selected_yolo(self):
        for item in self.yolo_viewer.scene.selectedItems():
            if hasattr(self.yolo_viewer, 'yolo_items') and item in self.yolo_viewer.yolo_items:
                self.yolo_viewer.scene.removeItem(item)
                self.yolo_viewer.yolo_items.remove(item)

    def clear_all_yolo(self):
        if hasattr(self.yolo_viewer, 'yolo_items'):
            for item in self.yolo_viewer.yolo_items[:]:
                self.yolo_viewer.scene.removeItem(item)
            self.yolo_viewer.yolo_items.clear()

    def delete_current_yolo_file(self):
        """Xóa vĩnh viễn file ảnh và file nhãn (.txt) của file đang xem trong Tab 3"""
        if not self.current_yolo_img_path or not os.path.exists(self.current_yolo_img_path):
            QMessageBox.information(self, "Thông báo", "Không có file ảnh nào đang được mở để xóa!")
            return

        img_path = os.path.abspath(self.current_yolo_img_path)
        img_dir = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Thu thập các đường dẫn file nhãn tương ứng có thể có
        candidate_lbls = set()
        if self.current_yolo_txt_path:
            candidate_lbls.add(os.path.abspath(self.current_yolo_txt_path))
        if self.current_yolo_folder:
            candidate_lbls.add(os.path.abspath(os.path.join(self.current_yolo_folder, f"{base_name}.txt")))
            candidate_lbls.add(os.path.abspath(os.path.join(self.current_yolo_folder, "labels", f"{base_name}.txt")))
        candidate_lbls.add(os.path.abspath(os.path.join(img_dir, f"{base_name}.txt")))
        candidate_lbls.add(os.path.abspath(os.path.join(img_dir, "labels", f"{base_name}.txt")))
        parent_dir = os.path.dirname(img_dir)
        candidate_lbls.add(os.path.abspath(os.path.join(parent_dir, "labels", f"{base_name}.txt")))

        existing_lbls = [p for p in candidate_lbls if os.path.isfile(p)]

        if not getattr(self, '_skip_confirm_delete_yolo_file', False):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Xác nhận xóa file ảnh & nhãn")
            lbl_info = "\n".join([f"• Nhãn: {os.path.basename(p)}" for p in existing_lbls]) if existing_lbls else "• Nhãn: (Không có file nhãn)"
            msg_box.setText(
                f"Bạn có chắc chắn muốn xóa vĩnh viễn file đang xem khỏi đĩa?\n\n"
                f"• Ảnh: {os.path.basename(img_path)}\n"
                f"{lbl_info}"
            )
            msg_box.setIcon(QMessageBox.Icon.Question)
            btn_yes = msg_box.addButton("Xóa vĩnh viễn", QMessageBox.ButtonRole.YesRole)
            btn_no = msg_box.addButton("Hủy bỏ", QMessageBox.ButtonRole.NoRole)
            msg_box.setDefaultButton(btn_no)

            chk_dont_ask = QCheckBox("Không hỏi lại trong phiên làm việc này")
            msg_box.setCheckBox(chk_dont_ask)

            msg_box.exec()

            if msg_box.clickedButton() != btn_yes:
                return

            if chk_dont_ask.isChecked():
                self._skip_confirm_delete_yolo_file = True

        # Tiến hành xóa ảnh
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xóa ảnh", f"Không thể xóa file ảnh:\n{img_path}\n\nLỗi: {e}")
            return

        # Tiến hành xóa nhãn (nếu có)
        for lbl_path in existing_lbls:
            try:
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
            except Exception as e:
                print(f"Warning: Không thể xóa nhãn {lbl_path}: {e}")

        # Xóa item khỏi danh sách giao diện
        cur_row = self.list_yolo_images.currentRow()
        if cur_row < 0:
            for r in range(self.list_yolo_images.count()):
                it = self.list_yolo_images.item(r)
                if it and it.text().replace("✅ ", "") == os.path.basename(img_path):
                    cur_row = r
                    break

        if cur_row >= 0:
            self.list_yolo_images.blockSignals(True)
            taken = self.list_yolo_images.takeItem(cur_row)
            del taken
            self.list_yolo_images.blockSignals(False)

        count = self.list_yolo_images.count()
        if count > 0:
            next_row = cur_row if cur_row < count else count - 1
            self.list_yolo_images.setCurrentRow(next_row)
            next_item = self.list_yolo_images.currentItem()
            if next_item:
                self.on_yolo_image_selected(next_item)
        else:
            self.current_yolo_img_path = ""
            self.current_yolo_txt_path = ""
            self.clear_all_yolo()
            self.yolo_viewer.scene.clear()
            self.yolo_viewer.image_item = None
            if hasattr(self.yolo_viewer, 'current_cv_img'):
                self.yolo_viewer.current_cv_img = None

    def browse_yolo_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa Ảnh và Nhãn YOLO")
        if path:
            self.current_yolo_folder = path
            self.list_yolo_images.clear()

            # Tự động phát hiện kpt_shape từ data.yaml nếu có trong thư mục dataset
            for candidate in [os.path.join(path, "data.yaml"), os.path.join(os.path.dirname(path), "data.yaml"), os.path.join(path, "..", "data.yaml")]:
                if os.path.exists(candidate):
                    try:
                        with open(candidate, 'r', encoding='utf-8', errors='ignore') as yf:
                            for yl in yf:
                                if 'kpt_shape' in yl:
                                    import re
                                    m = re.search(r'\[\s*(\d+)', yl)
                                    if m and hasattr(self, 'spn_pose_kpts'):
                                        kpts_count = int(m.group(1))
                                        self.spn_pose_kpts.blockSignals(True)
                                        self.spn_pose_kpts.setValue(kpts_count)
                                        self.spn_pose_kpts.blockSignals(False)
                                        self.yolo_viewer.num_pose_keypoints = kpts_count
                                        break
                    except Exception:
                        pass

            files = self._get_image_files(path)
            for f in files:
                name = os.path.basename(f)
                base = os.path.splitext(name)[0]
                txt_path = os.path.join(path, f"{base}.txt")
                if not os.path.exists(txt_path):
                    txt_path = os.path.join(path, "labels", f"{base}.txt")
                
                if os.path.exists(txt_path):
                    self.list_yolo_images.addItem(f"✅ {name}")
                else:
                    self.list_yolo_images.addItem(name)

    def on_yolo_image_changed(self, current, previous):
        if current is not None:
            self.on_yolo_image_selected(current)

    def on_yolo_image_selected(self, item):
        try:
            name = item.text().replace("✅ ", "")
            self.current_yolo_img_path = os.path.join(self.current_yolo_folder, name)
            
            img = cv2_imread(self.current_yolo_img_path)
            if img is None:
                return
                
            self.yolo_viewer.load_image_from_array(img)
            
            base = os.path.splitext(name)[0]
            txt_path = os.path.join(self.current_yolo_folder, f"{base}.txt")
            if not os.path.exists(txt_path):
                txt_path = os.path.join(self.current_yolo_folder, "labels", f"{base}.txt")
                
            self.current_yolo_txt_path = txt_path
            
            if os.path.exists(txt_path):
                labels = []
                detected_pose_kpts = None
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        parsed = parse_yolo_line(line_str)
                        if parsed:
                            labels.append(parsed)
                            if parsed['type'] == 'pose':
                                detected_pose_kpts = (len(parsed['data']) - 5) // 3

                # Tự động đồng bộ số lượng Kpts trên Tab 3 khớp với file nhãn vừa load
                if detected_pose_kpts is not None and hasattr(self, 'spn_pose_kpts'):
                    self.spn_pose_kpts.blockSignals(True)
                    self.spn_pose_kpts.setValue(detected_pose_kpts)
                    self.spn_pose_kpts.blockSignals(False)
                    self.yolo_viewer.num_pose_keypoints = detected_pose_kpts
                
                h, w = img.shape[:2]
                self.yolo_viewer.draw_yolo_labels(labels, w, h)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Lỗi khi đọc file nhãn: {e}", 4000)

    def save_yolo_labels(self):
        if not self.current_yolo_txt_path or not self.yolo_viewer.image_item:
            QMessageBox.warning(self, "Lỗi", "Chưa mở ảnh nào hoặc không có file label!")
            return
            
        rect = self.yolo_viewer.image_item.boundingRect()
        img_w = rect.width()
        img_h = rect.height()
        if img_w <= 0 or img_h <= 0:
            QMessageBox.warning(self, "Lỗi", "Kích thước ảnh không hợp lệ (img_w hoặc img_h <= 0)!")
            return
        
        lines = []
        for item in self.yolo_viewer.yolo_items:
            c_id = item.class_id if hasattr(item, 'class_id') and item.class_id is not None else 0
            
            if hasattr(item, 'item_type') and item.item_type == "YOLO_LABEL_POLY":
                poly = item.polygon()
                if poly.size() < 3:
                    continue
                p = item.pos()
                points_str = []
                for i in range(poly.size()):
                    pt = poly.at(i)
                    abs_x = pt.x() + p.x()
                    abs_y = pt.y() + p.y()
                    norm_x = max(0.0, min(1.0, abs_x / img_w))
                    norm_y = max(0.0, min(1.0, abs_y / img_h))
                    points_str.append(f"{norm_x:.6f} {norm_y:.6f}")
                lines.append(f"{c_id} " + " ".join(points_str))
            elif hasattr(item, 'item_type') and item.item_type == "YOLO_LABEL_POSE":
                r = item.rect()
                p = item.pos()
                x = r.x() + p.x()
                y = r.y() + p.y()
                w = r.width()
                h = r.height()
                
                x_center = max(0.0, min(1.0, (x + w / 2) / img_w))
                y_center = max(0.0, min(1.0, (y + h / 2) / img_h))
                w_norm = max(0.0, min(1.0, w / img_w))
                h_norm = max(0.0, min(1.0, h / img_h))
                
                parts_str = [f"{c_id}", f"{x_center:.6f}", f"{y_center:.6f}", f"{w_norm:.6f}", f"{h_norm:.6f}"]
                
                kpts = getattr(item, 'keypoints', [])
                num_kpts = max(len(kpts), self.spn_pose_kpts.value())
                for i in range(num_kpts):
                    if i < len(kpts):
                        scene_pt = item.mapToScene(kpts[i]['pos'])
                        kpt_x = max(0.0, min(1.0, scene_pt.x() / img_w))
                        kpt_y = max(0.0, min(1.0, scene_pt.y() / img_h))
                        v = kpts[i]['v']
                        parts_str.extend([f"{kpt_x:.6f}", f"{kpt_y:.6f}", f"{v}"])
                    else:
                        parts_str.extend(["0.000000", "0.000000", "0"])
                
                lines.append(" ".join(parts_str))
            else:
                r = item.rect()
                p = item.pos()
                x = r.x() + p.x()
                y = r.y() + p.y()
                w = r.width()
                h = r.height()
                
                # Đưa về YOLO format
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # clamp [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                lines.append(f"{c_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        os.makedirs(os.path.dirname(self.current_yolo_txt_path), exist_ok=True)
        with open(self.current_yolo_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            
        # Cập nhật trạng thái không chặn popup
        self.statusBar().showMessage(f"💾 Đã lưu {len(lines)} nhãn vào {os.path.basename(self.current_yolo_txt_path)}", 3000)
        
        curr_row = self.list_yolo_images.currentRow()
        if curr_row >= 0:
            item_obj = self.list_yolo_images.item(curr_row)
            if not item_obj.text().startswith("✅ "):
                item_obj.setText(f"✅ {item_obj.text()}")

    # ==========================================
    # TAB 4: AUTO PRS & YOLO-SEG
    # ==========================================
    def setup_tab_auto_prs(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ==========================================
        # LEFT PANEL: Cài đặt & Điều khiển (Sidebar)
        # ==========================================
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(2, 2, 2, 2)
        left_lay.setSpacing(6)

        # Tab Widget phân nhóm cấu hình thành 2 tab gọn gàng
        self.prs_settings_tabs = QTabWidget()
        self.prs_settings_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #45475a; border-radius: 6px; }
            QTabBar::tab { padding: 6px 10px; font-weight: bold; font-size: 11px; }
            QTabBar::tab:selected { background: #313244; color: #89b4fa; border-bottom: 2px solid #89b4fa; }
        """)

        # ------------------------------------------
        # SUB-TAB 1: Dữ liệu & PRS (Tab 1)
        # ------------------------------------------
        tab_data = QWidget()
        lay_data = QVBoxLayout(tab_data)
        lay_data.setContentsMargins(4, 6, 4, 6)
        lay_data.setSpacing(8)

        # 1. Thư mục dữ liệu
        grp_folders = QGroupBox("📁 Thư mục Dữ liệu")
        lay_folders = QGridLayout()
        lay_folders.setSpacing(5)

        lay_folders.addWidget(QLabel("📂 Input Images:"), 0, 0)
        self.txt_prs_input_folder = QLineEdit()
        self.txt_prs_input_folder.setPlaceholderText("Thư mục chứa ảnh gốc cần dò...")
        lay_folders.addWidget(self.txt_prs_input_folder, 0, 1)
        btn_prs_in = QPushButton("...")
        btn_prs_in.setFixedWidth(36)
        btn_prs_in.clicked.connect(self.browse_prs_input)
        lay_folders.addWidget(btn_prs_in, 0, 2)

        lay_folders.addWidget(QLabel("💾 Output Dataset:"), 1, 0)
        self.txt_prs_output_folder = QLineEdit()
        self.txt_prs_output_folder.setPlaceholderText("Thư mục lưu nhãn / dataset...")
        lay_folders.addWidget(self.txt_prs_output_folder, 1, 1)
        btn_prs_out = QPushButton("...")
        btn_prs_out.setFixedWidth(36)
        btn_prs_out.clicked.connect(self.browse_prs_output)
        lay_folders.addWidget(btn_prs_out, 1, 2)

        self.chk_prs_save_images = QCheckBox("Lưu bản sao ảnh sang thư mục images/")
        self.chk_prs_save_images.setChecked(False)
        lay_folders.addWidget(self.chk_prs_save_images, 2, 0, 1, 3)

        grp_folders.setLayout(lay_folders)
        lay_data.addWidget(grp_folders)

        # 2. PRS & Defect Zone (từ Tab 1)
        grp_prs = QGroupBox("🔍 PRS & Defect Zone (Tab 1)")
        lay_prs = QVBoxLayout()
        lay_prs.setSpacing(6)

        lbl_desc = QLabel(
            "📌 Dò PRS định vị → Ánh xạ Defect Zone → Tự động tạo nhãn YOLO-Seg / Pose."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #89b4fa; font-size: 11px;")
        lay_prs.addWidget(lbl_desc)

        lay_status_row = QHBoxLayout()
        self.lbl_prs_status = QLabel("Pattern 1: ❓ | Pattern 2: ❓")
        self.lbl_prs_status.setStyleSheet("font-weight: bold; color: #fab387; font-size: 11px;")
        lay_status_row.addWidget(self.lbl_prs_status)
        lay_status_row.addStretch()
        self.lbl_zones_count = QLabel("Defect Zones: 0")
        self.lbl_zones_count.setStyleSheet("color: #a6adc8; font-size: 11px;")
        lay_status_row.addWidget(self.lbl_zones_count)
        lay_prs.addLayout(lay_status_row)

        lay_margin = QHBoxLayout()
        lay_margin.addWidget(QLabel("Mở rộng ROI (Margin):"))
        self.spn_prs_margin = QSpinBox()
        self.spn_prs_margin.setRange(0, 200)
        self.spn_prs_margin.setValue(5)
        self.spn_prs_margin.setSuffix(" px")
        self.spn_prs_margin.setToolTip("Mở rộng thêm xung quanh vùng Defect Zone để đảm bảo bao trọn đối tượng")
        lay_margin.addWidget(self.spn_prs_margin)
        lay_prs.addLayout(lay_margin)

        self.chk_prs_zone_as_seg = QCheckBox("⭐ Đường Defect Zone làm nhãn Seg luôn")
        self.chk_prs_zone_as_seg.setStyleSheet("font-weight: bold; color: #a6e3a1; font-size: 11px;")
        self.chk_prs_zone_as_seg.setToolTip(
            "Tích chọn: Dùng trực tiếp đường viền Defect Zone (đã căn chỉnh theo PRS) làm nhãn segmentation,\n"
            "bỏ qua bước phân vùng nhị phân (Binary Threshold)."
        )
        self.chk_prs_zone_as_seg.toggled.connect(self.on_prs_zone_as_seg_toggled)
        lay_prs.addWidget(self.chk_prs_zone_as_seg)

        grp_prs.setLayout(lay_prs)
        lay_data.addWidget(grp_prs)

        # 3. Danh sách ảnh
        grp_img_list = QGroupBox("📋 Danh sách file ảnh (chọn để thử preview)")
        lay_img_list = QVBoxLayout()
        lay_img_list.setSpacing(4)
        self.list_prs_images = QListWidget()
        self.list_prs_images.setMinimumHeight(140)
        self.list_prs_images.currentItemChanged.connect(self.on_prs_image_changed)
        lay_img_list.addWidget(self.list_prs_images)
        grp_img_list.setLayout(lay_img_list)
        lay_data.addWidget(grp_img_list)

        lay_data.addStretch()

        scroll_data = QScrollArea()
        scroll_data.setWidgetResizable(True)
        scroll_data.setWidget(tab_data)
        scroll_data.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.prs_settings_tabs.addTab(scroll_data, "📁 Dữ liệu & PRS")

        # ------------------------------------------
        # SUB-TAB 2: Cấu hình Nhãn Tự động
        # ------------------------------------------
        tab_label_cfg = QWidget()
        lay_label_cfg = QVBoxLayout(tab_label_cfg)
        lay_label_cfg.setContentsMargins(4, 6, 4, 6)
        lay_label_cfg.setSpacing(6)

        grp_seg = QGroupBox("🪄 Cấu hình Nhãn Tự động")
        lay_grp3 = QVBoxLayout()
        lay_grp3.setSpacing(6)

        # Row 1: Chọn Class & các nút thêm/xóa class
        lay_cls_row = QHBoxLayout()
        lay_cls_row.setSpacing(6)
        lay_cls_row.addWidget(QLabel("🏷 Chọn Class:"))
        self.cmb_prs_class = QComboBox()
        self.cmb_prs_class.setMinimumHeight(28)
        self.cmb_prs_class.setStyleSheet("font-size: 12px; font-weight: bold;")
        lay_cls_row.addWidget(self.cmb_prs_class, stretch=1)

        btn_add_c = QPushButton("➕ Thêm")
        btn_add_c.setStyleSheet("QPushButton { padding: 4px 10px; font-size: 11px; min-height: 22px; }")
        btn_add_c.clicked.connect(self.add_prs_class)
        btn_del_c = QPushButton("❌ Xóa")
        btn_del_c.setStyleSheet("QPushButton { padding: 4px 10px; font-size: 11px; min-height: 22px; }")
        btn_del_c.clicked.connect(self.del_prs_class)
        btn_clr_c = QPushButton("🧹 Xóa hết")
        btn_clr_c.setStyleSheet("QPushButton { padding: 4px 10px; font-size: 11px; min-height: 22px; }")
        btn_clr_c.clicked.connect(self.clear_prs_classes)
        lay_cls_row.addWidget(btn_add_c)
        lay_cls_row.addWidget(btn_del_c)
        lay_cls_row.addWidget(btn_clr_c)
        lay_grp3.addLayout(lay_cls_row)

        # Row 2: Chọn kiểu nhãn YOLO (Radio buttons)
        lay_mode = QHBoxLayout()
        lay_mode.setContentsMargins(0, 8, 0, 8)
        lay_mode.setSpacing(12)
        lbl_mode = QLabel("🎯 Kiểu nhãn:")
        lbl_mode.setStyleSheet("font-weight: bold;")
        lay_mode.addWidget(lbl_mode)
        self.rb_prs_mode_seg = QRadioButton("YOLO-Seg (Phân vùng)")
        self.rb_prs_mode_pose = QRadioButton("YOLO-Pose (Khung & Keypoints)")
        self.rb_prs_mode_seg.setStyleSheet("font-weight: bold; color: #a6e3a1; font-size: 12px;")
        self.rb_prs_mode_pose.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        self.rb_prs_mode_seg.setChecked(True)
        self.prs_mode_group = QButtonGroup(self)
        self.prs_mode_group.addButton(self.rb_prs_mode_seg, 0)
        self.prs_mode_group.addButton(self.rb_prs_mode_pose, 1)
        self.rb_prs_mode_seg.toggled.connect(self.on_prs_label_mode_changed)
        lay_mode.addWidget(self.rb_prs_mode_seg)
        lay_mode.addWidget(self.rb_prs_mode_pose)
        lay_mode.addStretch()
        lay_grp3.addLayout(lay_mode)

        # Widget Cấu hình YOLO-Seg
        self.widget_seg_settings = QWidget()
        lay_seg = QGridLayout(self.widget_seg_settings)
        lay_seg.setContentsMargins(0, 4, 0, 0)
        lay_seg.setSpacing(5)

        self.chk_prs_otsu = QCheckBox("Tự động ngưỡng (Otsu)")
        self.chk_prs_otsu.setChecked(False)
        self.chk_prs_otsu.toggled.connect(self.on_prs_otsu_toggled)
        lay_seg.addWidget(self.chk_prs_otsu, 0, 0, 1, 3)

        self.lbl_prs_th = QLabel("Ngưỡng:")
        lay_seg.addWidget(self.lbl_prs_th, 1, 0)
        self.slider_prs_th = QSlider(Qt.Orientation.Horizontal)
        self.slider_prs_th.setRange(0, 255)
        self.slider_prs_th.setValue(128)
        lay_seg.addWidget(self.slider_prs_th, 1, 1)

        self.spn_prs_th = QSpinBox()
        self.spn_prs_th.setRange(0, 255)
        self.spn_prs_th.setValue(128)
        self.spn_prs_th.setFixedWidth(55)
        lay_seg.addWidget(self.spn_prs_th, 1, 2)

        self.slider_prs_th.valueChanged.connect(self.spn_prs_th.setValue)
        self.spn_prs_th.valueChanged.connect(self.slider_prs_th.setValue)
        self.spn_prs_th.valueChanged.connect(self.on_prs_seg_opts_changed)

        lay_seg.addWidget(QLabel("Min Area:"), 2, 0)
        self.spn_prs_min_area = QSpinBox()
        self.spn_prs_min_area.setRange(1, 100000)
        self.spn_prs_min_area.setValue(10)
        self.spn_prs_min_area.setSuffix(" px")
        self.spn_prs_min_area.valueChanged.connect(self.on_prs_seg_opts_changed)
        lay_seg.addWidget(self.spn_prs_min_area, 2, 1, 1, 2)

        lay_seg.addWidget(QLabel("Độ mượt (Epsilon):"), 3, 0)
        self.spn_prs_epsilon = QDoubleSpinBox()
        self.spn_prs_epsilon.setRange(0.0005, 0.05)
        self.spn_prs_epsilon.setValue(0.005)
        self.spn_prs_epsilon.setSingleStep(0.001)
        self.spn_prs_epsilon.setDecimals(4)
        self.spn_prs_epsilon.setToolTip("Tỉ lệ xấp xỉ approxPolyDP. Nhỏ giữ nhiều điểm, lớn giảm số đỉnh")
        self.spn_prs_epsilon.valueChanged.connect(self.on_prs_seg_opts_changed)
        lay_seg.addWidget(self.spn_prs_epsilon, 3, 1, 1, 2)

        # Dò biên: Polarity radio buttons
        lay_seg.addWidget(QLabel("Dò biên:"), 4, 0)
        self.rb_pol_b2w = QRadioButton("Đen → Trắng")
        self.rb_pol_w2b = QRadioButton("Trắng → Đen")
        self.rb_pol_any = QRadioButton("Bất kỳ")
        self.rb_pol_b2w.setChecked(True)
        self.pol_group = QButtonGroup(self)
        self.pol_group.addButton(self.rb_pol_b2w, 0)
        self.pol_group.addButton(self.rb_pol_w2b, 1)
        self.pol_group.addButton(self.rb_pol_any, 2)
        lay_pol = QHBoxLayout()
        lay_pol.setContentsMargins(0, 0, 0, 0)
        lay_pol.addWidget(self.rb_pol_b2w)
        lay_pol.addWidget(self.rb_pol_w2b)
        lay_pol.addWidget(self.rb_pol_any)
        pol_widget = QWidget()
        pol_widget.setLayout(lay_pol)
        lay_seg.addWidget(pol_widget, 4, 1, 1, 2)
        self.pol_group.idToggled.connect(lambda _id, _chk: self.on_prs_seg_opts_changed())

        # Hướng tìm kiếm
        lay_seg.addWidget(QLabel("Hướng tìm:"), 5, 0)
        self.cmb_search_dir = QComboBox()
        self.cmb_search_dir.addItems([
            "Ngoài → Trong (Outside-In)",
            "Trong → Ngoài (Inside-Out)"
        ])
        self.cmb_search_dir.currentIndexChanged.connect(self.on_prs_seg_opts_changed)
        lay_seg.addWidget(self.cmb_search_dir, 5, 1, 1, 2)

        # Khoảng cách tìm kiếm tối đa
        lay_seg.addWidget(QLabel("Khoảng cách:"), 6, 0)
        self.spn_search_dist = QSpinBox()
        self.spn_search_dist.setRange(0, 5000)
        self.spn_search_dist.setValue(100)
        self.spn_search_dist.setSingleStep(10)
        self.spn_search_dist.setSuffix(" px")
        self.spn_search_dist.setSpecialValueText("Không giới hạn")
        self.spn_search_dist.setToolTip("Khoảng cách tối đa để quét tìm biên từ điểm bắt đầu")
        self.spn_search_dist.valueChanged.connect(self.on_prs_seg_opts_changed)
        lay_seg.addWidget(self.spn_search_dist, 6, 1, 1, 2)

        self.chk_dzone_fallback = QCheckBox("Không đổi màu: Nhận biên Defect Zone")
        self.chk_dzone_fallback.setChecked(True)
        self.chk_dzone_fallback.setToolTip("Khi tia dò không thấy chuyển mức màu: lấy biên Defect Zone làm segment")
        self.chk_dzone_fallback.toggled.connect(self.on_prs_seg_opts_changed)
        lay_seg.addWidget(self.chk_dzone_fallback, 7, 0, 1, 3)

        lay_grp3.addWidget(self.widget_seg_settings)

        # Widget Cấu hình YOLO-Pose
        self.widget_pose_settings = QWidget()
        lay_pose = QGridLayout(self.widget_pose_settings)
        lay_pose.setContentsMargins(0, 4, 0, 0)
        lay_pose.setSpacing(5)

        lay_pose.addWidget(QLabel("Chấm điểm:"), 0, 0)
        self.cmb_pose_kpt_mode = QComboBox()
        self.cmb_pose_kpt_mode.addItem("📍 Dùng điểm chấm trên Master (Tab 1)", "from_tab1")
        self.cmb_pose_kpt_mode.addItem("2 điểm: 2 đầu mút trục chính (PCA)", "two_points")
        self.cmb_pose_kpt_mode.addItem("4 điểm: 4 góc của Defect Zone", "corners")
        self.cmb_pose_kpt_mode.addItem("1 điểm: Tâm Defect Zone (Centroid)", "center")
        self.cmb_pose_kpt_mode.addItem("N điểm: Phân bố đều dọc trục chính", "equidistant")
        self.cmb_pose_kpt_mode.addItem("Tâm Blob khuyết tật (Binary Blob)", "centroid_blob")
        self.cmb_pose_kpt_mode.currentIndexChanged.connect(self.on_pose_kpt_mode_changed)
        lay_pose.addWidget(self.cmb_pose_kpt_mode, 0, 1)

        lay_pose.addWidget(QLabel("Số Keypoint:"), 1, 0)
        self.spn_pose_num_kpts = QSpinBox()
        self.spn_pose_num_kpts.setRange(1, 99)
        self.spn_pose_num_kpts.setValue(2)
        self.spn_pose_num_kpts.setSuffix(" điểm")
        self.spn_pose_num_kpts.setEnabled(False)
        lay_pose.addWidget(self.spn_pose_num_kpts, 1, 1)

        lbl_pose_info = QLabel("💡 Điểm chấm được thiết lập ở Tab 1. BBox là Defect Zone và Keypoints biến đổi theo PRS.")
        lbl_pose_info.setWordWrap(True)
        lbl_pose_info.setStyleSheet("color: #89b4fa; font-size: 11px; padding: 2px;")
        lay_pose.addWidget(lbl_pose_info, 2, 0, 1, 2)

        self.widget_pose_settings.setVisible(False)
        lay_grp3.addWidget(self.widget_pose_settings)

        grp_seg.setLayout(lay_grp3)
        lay_label_cfg.addWidget(grp_seg)
        lay_label_cfg.addStretch()

        scroll_label_cfg = QScrollArea()
        scroll_label_cfg.setWidgetResizable(True)
        scroll_label_cfg.setWidget(tab_label_cfg)
        scroll_label_cfg.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.prs_settings_tabs.addTab(scroll_label_cfg, "🪄 Cấu hình Nhãn")

        left_lay.addWidget(self.prs_settings_tabs)

        # Bottom of left sidebar: Actions & Progress
        self.btn_prs_preview = QPushButton("👁 Thử 1 ảnh (Preview)")
        self.btn_prs_preview.setMinimumHeight(36)
        self.btn_prs_preview.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #a855f7); "
            "color: white; font-weight: bold; border-radius: 6px; font-size: 12px; border: none; }"
            "QPushButton:hover { background: #a78bfa; }"
        )
        self.btn_prs_preview.clicked.connect(self.preview_single_prs_seg)
        left_lay.addWidget(self.btn_prs_preview)

        lay_run_btns = QHBoxLayout()
        self.btn_run_prs = QPushButton("🚀 CHẠY TẤT CẢ")
        self.btn_run_prs.setMinimumHeight(44)
        self.btn_run_prs.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #4CAF50, stop:1 #66BB6A);
                          color: white; font-weight: bold; font-size: 14px;
                          border: none; border-radius: 8px; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                stop:0 #66BB6A, stop:1 #81C784); }
        """)
        self.btn_run_prs.clicked.connect(self.toggle_run_prs)
        lay_run_btns.addWidget(self.btn_run_prs, stretch=3)

        self.btn_stop_prs = QPushButton("⏹ Dừng")
        self.btn_stop_prs.setMinimumHeight(44)
        self.btn_stop_prs.setEnabled(False)
        self.btn_stop_prs.setStyleSheet("""
            QPushButton { background: #e74c3c; color: white; font-weight: bold; font-size: 13px;
                          border: none; border-radius: 8px; }
            QPushButton:hover { background: #c0392b; }
            QPushButton:disabled { background: #555555; color: #888888; }
        """)
        self.btn_stop_prs.clicked.connect(self.stop_run_prs)
        lay_run_btns.addWidget(self.btn_stop_prs, stretch=1)
        left_lay.addLayout(lay_run_btns)

        self.prs_progress_bar = QProgressBar()
        self.prs_progress_bar.setValue(0)
        self.prs_progress_bar.setMaximumHeight(20)
        left_lay.addWidget(self.prs_progress_bar)

        main_splitter.addWidget(left_widget)

        # ==========================================
        # RIGHT PANEL: Preview Viewer & Log
        # ==========================================
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(2, 2, 2, 2)
        right_lay.setSpacing(4)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top area of right side: Header bar + ImageViewer
        viewer_container = QWidget()
        viewer_lay = QVBoxLayout(viewer_container)
        viewer_lay.setContentsMargins(0, 0, 0, 0)
        viewer_lay.setSpacing(4)

        viewer_header = QHBoxLayout()
        lbl_viewer_title = QLabel("🖼️ Vùng hiển thị & Kiểm tra nhãn tự động (Review)")
        lbl_viewer_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 13px;")
        viewer_header.addWidget(lbl_viewer_title)
        viewer_header.addStretch()

        btn_toggle_log = QPushButton("📟 Ẩn/Hiện Log")
        btn_toggle_log.setCheckable(True)
        btn_toggle_log.setChecked(True)
        btn_toggle_log.setFixedHeight(26)
        btn_toggle_log.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        viewer_header.addWidget(btn_toggle_log)
        viewer_lay.addLayout(viewer_header)

        self.prs_viewer = ImageViewer()
        self.prs_viewer.current_tool = "NONE"
        viewer_lay.addWidget(self.prs_viewer)

        right_splitter.addWidget(viewer_container)

        # Bottom area of right side: Terminal Log
        log_widget = QWidget()
        log_lay = QVBoxLayout(log_widget)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(2)
        lbl_log = QLabel("📟 Log tiến trình Auto PRS & Labeling:")
        lbl_log.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        log_lay.addWidget(lbl_log)
        self.txt_prs_log = QTextEdit()
        self.txt_prs_log.setReadOnly(True)
        self.txt_prs_log.setMaximumHeight(200)
        log_lay.addWidget(self.txt_prs_log)

        right_splitter.addWidget(log_widget)

        btn_toggle_log.toggled.connect(log_widget.setVisible)

        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 1)

        right_lay.addWidget(right_splitter)
        main_splitter.addWidget(right_widget)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([700, 700])

        main_layout.addWidget(main_splitter)
        self.tab_auto_prs.setLayout(main_layout)

    def browse_prs_input(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục Ảnh Input")
        if path:
            self.txt_prs_input_folder.setText(path)
            self.current_prs_input_folder = path
            self.refresh_prs_images_list()

    def browse_prs_output(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục Dataset Output")
        if path:
            self.txt_prs_output_folder.setText(path)
            self.current_prs_output_folder = path

    def refresh_prs_images_list(self):
        folder = self.txt_prs_input_folder.text().strip()
        self.list_prs_images.clear()
        if not folder or not os.path.exists(folder):
            return
        files = self._get_image_files(folder)
        out_folder = self.txt_prs_output_folder.text().strip()
        labels_folder = os.path.join(out_folder, "labels") if out_folder else ""
        for f in files:
            name = os.path.basename(f)
            base = os.path.splitext(name)[0]
            txt_path = os.path.join(labels_folder, f"{base}.txt") if labels_folder else ""
            if txt_path and os.path.exists(txt_path):
                self.list_prs_images.addItem(f"✅ {name}")
            else:
                self.list_prs_images.addItem(name)

    def refresh_auto_prs_tab(self):
        """Đồng bộ trạng thái Pattern và Defect Zones từ Tab 1 và danh sách Class"""
        self.sync_tab1_to_recipe()
        p1 = bool(self.recipe_data.get('prs1', {}).get('tpl_b64'))
        p2 = bool(self.recipe_data.get('prs2', {}).get('tpl_b64'))
        p1_txt = "✅ Đã nạp" if p1 else "❌ Chưa nạp"
        p2_txt = "✅ Đã nạp" if p2 else "❌ Chưa nạp"
        self.lbl_prs_status.setText(f"Pattern 1: {p1_txt}  |  Pattern 2: {p2_txt}")
        zones = self.recipe_data.get('defect_zones', [])
        ignore_zones = self.recipe_data.get('ignore_zones', [])
        total_kpts = sum(len(z.get('keypoints', [])) for z in zones)
        self.lbl_zones_count.setText(f"Defect Zones: {len(zones)} vùng (Pose: {total_kpts} điểm) | Ignore: {len(ignore_zones)} vùng")
        if hasattr(self, 'cmb_pose_kpt_mode') and self.cmb_pose_kpt_mode.currentData() == 'from_tab1':
            num_kpts = self.spn_tab1_pose_kpts.value() if hasattr(self, 'spn_tab1_pose_kpts') else 2
            self.spn_pose_num_kpts.setValue(num_kpts)
        self._sync_classes_to_prs()

    def _get_prs_selected_class_id(self):
        text = self.cmb_prs_class.currentText()
        if text:
            import re
            m = re.match(r"^(\d+)\s*[-:]\s*(.+)$", text)
            if m:
                return int(m.group(1))
        idx = self.cmb_prs_class.currentIndex()
        return idx if idx >= 0 else 0

    def on_prs_otsu_toggled(self, checked):
        self.slider_prs_th.setEnabled(not checked)
        self.spn_prs_th.setEnabled(not checked)
        self.on_prs_seg_opts_changed()

    def on_prs_zone_as_seg_toggled(self, checked):
        self.chk_prs_otsu.setEnabled(not checked)
        is_otsu = self.chk_prs_otsu.isChecked()
        self.lbl_prs_th.setEnabled(not checked and not is_otsu)
        self.slider_prs_th.setEnabled(not checked and not is_otsu)
        self.spn_prs_th.setEnabled(not checked and not is_otsu)
        self.spn_prs_min_area.setEnabled(not checked)
        self.spn_prs_epsilon.setEnabled(not checked)
        # Dò biên controls
        for rb in [self.rb_pol_b2w, self.rb_pol_w2b, self.rb_pol_any]:
            rb.setEnabled(not checked)
        self.cmb_search_dir.setEnabled(not checked)
        self.spn_search_dist.setEnabled(not checked)
        self.chk_dzone_fallback.setEnabled(not checked)
        self.on_prs_seg_opts_changed()

    def on_prs_seg_opts_changed(self):
        pass

    def on_prs_label_mode_changed(self):
        is_pose = self.rb_prs_mode_pose.isChecked()
        self.widget_seg_settings.setVisible(not is_pose)
        self.widget_pose_settings.setVisible(is_pose)
        if is_pose:
            self.chk_prs_zone_as_seg.setEnabled(False)
            self.btn_prs_preview.setText("👁 Thử 1 ảnh (YOLO-Pose)")
        else:
            self.chk_prs_zone_as_seg.setEnabled(True)
            self.on_prs_zone_as_seg_toggled(self.chk_prs_zone_as_seg.isChecked())
            self.btn_prs_preview.setText("👁 Thử 1 ảnh (YOLO-Seg)")

    def on_pose_kpt_mode_changed(self, index):
        mode_data = self.cmb_pose_kpt_mode.currentData()
        if mode_data == 'from_tab1':
            num = self.spn_tab1_pose_kpts.value() if hasattr(self, 'spn_tab1_pose_kpts') else 2
            self.spn_pose_num_kpts.setValue(num)
            self.spn_pose_num_kpts.setEnabled(False)
        elif mode_data == 'center':
            self.spn_pose_num_kpts.setValue(1)
            self.spn_pose_num_kpts.setEnabled(False)
        elif mode_data == 'two_points':
            self.spn_pose_num_kpts.setValue(2)
            self.spn_pose_num_kpts.setEnabled(False)
        elif mode_data == 'corners':
            self.spn_pose_num_kpts.setValue(4)
            self.spn_pose_num_kpts.setEnabled(False)
        elif mode_data in ('equidistant', 'centroid_blob'):
            self.spn_pose_num_kpts.setEnabled(True)

    def _get_pose_kpt_mode_key(self):
        data = self.cmb_pose_kpt_mode.currentData()
        return data if data else 'from_tab1'

    def on_prs_image_changed(self, current, previous):
        if current is not None:
            self.on_prs_image_selected(current)

    def on_prs_image_selected(self, item):
        name = item.text().replace("✅ ", "")
        folder = self.txt_prs_input_folder.text().strip()
        img_path = os.path.join(folder, name)
        if not os.path.exists(img_path):
            return
        self.current_prs_img_path = img_path

        out_folder = self.txt_prs_output_folder.text().strip()
        base = os.path.splitext(name)[0]
        txt_path = os.path.join(out_folder, "labels", f"{base}.txt") if out_folder else ""
        if not (txt_path and os.path.exists(txt_path)):
            txt_path = os.path.join(folder, f"{base}.txt")

        img = cv2_imread(img_path)
        if img is not None:
            self.prs_viewer.load_image_from_array(img)
            if hasattr(self, 'yolo_viewer'):
                self.prs_viewer.yolo_class_names = getattr(self.yolo_viewer, 'yolo_class_names', {})
            if os.path.exists(txt_path):
                labels = []
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parsed = parse_yolo_line(line)
                        if parsed:
                            labels.append(parsed)
                h, w = img.shape[:2]
                self.prs_viewer.draw_yolo_labels(labels, w, h)
            else:
                self.clear_prs_viewer_labels()

    def clear_prs_viewer_labels(self):
        if hasattr(self.prs_viewer, 'yolo_items'):
            for item in self.prs_viewer.yolo_items[:]:
                self.prs_viewer.scene.removeItem(item)
            self.prs_viewer.yolo_items.clear()

    def _get_auto_prs_config(self):
        self.sync_tab1_to_recipe()
        folder = self.txt_prs_input_folder.text().strip()
        out_folder = self.txt_prs_output_folder.text().strip()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục Ảnh Input hợp lệ!")
            return None
        if not out_folder:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục Dataset Output!")
            return None

        files = self._get_image_files(folder)
        if not files:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy file ảnh nào trong thư mục Input!")
            return None

        polarity_modes = ['black_to_white', 'white_to_black', 'any']
        pol_idx = self.pol_group.checkedId() if hasattr(self, 'pol_group') else 0
        polarity = polarity_modes[pol_idx] if 0 <= pol_idx < len(polarity_modes) else 'black_to_white'
        search_dir = 'outside_in' if self.cmb_search_dir.currentIndex() == 0 else 'inside_out'

        is_pose = self.rb_prs_mode_pose.isChecked()
        kpt_mode = self._get_pose_kpt_mode_key()
        num_kpts = self.spn_tab1_pose_kpts.value() if (kpt_mode == 'from_tab1' and hasattr(self, 'spn_tab1_pose_kpts')) else self.spn_pose_num_kpts.value()
        cfg = {
            'recipe': self.recipe_data,
            'input_files': files,
            'output_folder': out_folder,
            'labels_folder': os.path.join(out_folder, 'labels'),
            'save_images': self.chk_prs_save_images.isChecked(),
            'zone_as_seg': self.chk_prs_zone_as_seg.isChecked(),
            'label_mode': 'pose' if is_pose else 'seg',
            'pose_kpt_mode': kpt_mode,
            'num_kpts': num_kpts,
            'threshold': self.spn_prs_th.value(),
            'use_otsu': self.chk_prs_otsu.isChecked(),
            'min_area': self.spn_prs_min_area.value(),
            'epsilon_ratio': self.spn_prs_epsilon.value(),
            'class_id': self._get_prs_selected_class_id(),
            'classes': [self.cmb_prs_class.itemText(i) for i in range(self.cmb_prs_class.count())],
            'margin': self.spn_prs_margin.value(),
            'polarity': polarity,
            'search_direction': search_dir,
            'search_dist': self.spn_search_dist.value(),
            'dzone_fallback': self.chk_dzone_fallback.isChecked()
        }
        return cfg

    def preview_single_prs_seg(self):
        """Preview tức thì 1 ảnh: tìm PRS + Defect Zone + segment hoặc pose + hiển thị lên viewer"""
        try:
            self.sync_tab1_to_recipe()
            folder = self.txt_prs_input_folder.text().strip()
            if not folder or not os.path.exists(folder):
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục Ảnh Input trước!")
                return

            curr_item = self.list_prs_images.currentItem()
            if curr_item:
                name = curr_item.text().replace("✅ ", "")
                img_path = os.path.join(folder, name)
            else:
                files = self._get_image_files(folder)
                if not files:
                    QMessageBox.warning(self, "Lỗi", "Không có file ảnh nào trong thư mục Input!")
                    return
                img_path = files[0]

            polarity_modes = ['black_to_white', 'white_to_black', 'any']
            pol_idx = self.pol_group.checkedId() if hasattr(self, 'pol_group') else 0
            polarity = polarity_modes[pol_idx] if 0 <= pol_idx < len(polarity_modes) else 'black_to_white'
            search_dir = 'outside_in' if self.cmb_search_dir.currentIndex() == 0 else 'inside_out'

            is_pose = self.rb_prs_mode_pose.isChecked()
            kpt_mode = self._get_pose_kpt_mode_key()
            num_kpts = self.spn_tab1_pose_kpts.value() if (kpt_mode == 'from_tab1' and hasattr(self, 'spn_tab1_pose_kpts')) else self.spn_pose_num_kpts.value()
            cfg = {
                'recipe': self.recipe_data,
                'zone_as_seg': self.chk_prs_zone_as_seg.isChecked(),
                'label_mode': 'pose' if is_pose else 'seg',
                'pose_kpt_mode': kpt_mode,
                'num_kpts': num_kpts,
                'threshold': self.spn_prs_th.value(),
                'use_otsu': self.chk_prs_otsu.isChecked(),
                'min_area': self.spn_prs_min_area.value(),
                'epsilon_ratio': self.spn_prs_epsilon.value(),
                'class_id': self._get_prs_selected_class_id(),
                'classes': [self.cmb_prs_class.itemText(i) for i in range(self.cmb_prs_class.count())],
                'margin': self.spn_prs_margin.value(),
                'polarity': polarity,
                'search_direction': search_dir,
                'search_dist': self.spn_search_dist.value(),
                'dzone_fallback': self.chk_dzone_fallback.isChecked()
            }

            worker = AutoPrsSegWorkerThread(cfg)
            success, out_img, annots, polies = worker.process_single_image(img_path)
            if not success or out_img is None:
                QMessageBox.warning(self, "Thông báo", f"Không tìm thấy đối tượng trong ảnh:\n{os.path.basename(img_path)}")
                return

            self.prs_viewer.load_image_from_array(out_img)
            self.clear_prs_viewer_labels()
            if hasattr(self, 'yolo_viewer'):
                self.prs_viewer.yolo_class_names = getattr(self.yolo_viewer, 'yolo_class_names', {})

            cid = self._get_prs_selected_class_id()

            labels = []
            # Hiển thị các nhãn cũ của các class khác nếu file đã tồn tại
            out_folder = self.txt_prs_output_folder.text().strip()
            base = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(out_folder, "labels", f"{base}.txt") if out_folder else ""
            if txt_path and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parsed = parse_yolo_line(line)
                        if parsed and int(parsed['data'][0]) != cid:
                            labels.append(parsed)

            # Thêm các nhãn vừa tạo cho class hiện tại
            for line in annots:
                parsed = parse_yolo_line(line)
                if parsed:
                    labels.append(parsed)

            h, w = out_img.shape[:2]
            if labels:
                self.prs_viewer.draw_yolo_labels(labels, w, h)

            unit_str = "đối tượng YOLO-Pose" if is_pose else "đa giác segment"
            self.txt_prs_log.append(f"👁 Preview {os.path.basename(img_path)}: Phát hiện {len(annots)} {unit_str} (Class {cid}).")
            self.statusBar().showMessage(f"👁 Preview thành công: {len(annots)} {unit_str}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Preview", f"Đã xảy ra lỗi khi tạo preview:\n{e}")
            if hasattr(self, 'txt_prs_log'):
                self.txt_prs_log.append(f"❌ Lỗi Preview: {e}")

    def toggle_run_prs(self):
        cfg = self._get_auto_prs_config()
        if not cfg:
            return

        self.btn_run_prs.setEnabled(False)
        self.btn_stop_prs.setEnabled(True)
        self.btn_prs_preview.setEnabled(False)
        self.prs_progress_bar.setValue(0)
        self.txt_prs_log.clear()

        self.prs_worker = AutoPrsSegWorkerThread(cfg)
        self.prs_worker.log_signal.connect(self.on_prs_worker_log)
        self.prs_worker.progress_signal.connect(self.on_prs_worker_progress)
        self.prs_worker.preview_signal.connect(self.on_prs_worker_preview)
        self.prs_worker.finished_signal.connect(self.on_prs_worker_finished)
        self.prs_worker.start()

    def stop_run_prs(self):
        if self.prs_worker and self.prs_worker.isRunning():
            self.prs_worker.stop()
            self.txt_prs_log.append("⏹ Đang yêu cầu dừng tiến trình...")

    def on_prs_worker_log(self, msg):
        self.txt_prs_log.append(msg)
        self.txt_prs_log.moveCursor(QTextCursor.MoveOperation.End)

    def on_prs_worker_progress(self, current, total):
        pct = int(current / total * 100) if total > 0 else 0
        self.prs_progress_bar.setValue(pct)
        self.prs_progress_bar.setFormat(f"{current}/{total} ({pct}%)")

    def on_prs_worker_preview(self, out_img, filepath, polies):
        if out_img is not None:
            self.prs_viewer.load_image_from_array(out_img)
            self.clear_prs_viewer_labels()
            if hasattr(self, 'yolo_viewer'):
                self.prs_viewer.yolo_class_names = getattr(self.yolo_viewer, 'yolo_class_names', {})

            # Hiển thị tất cả các nhãn đã lưu (gồm các class khác đã gộp)
            out_folder = self.txt_prs_output_folder.text().strip()
            base = os.path.splitext(os.path.basename(filepath))[0]
            txt_path = os.path.join(out_folder, "labels", f"{base}.txt") if out_folder else ""
            h, w = out_img.shape[:2]
            if txt_path and os.path.exists(txt_path):
                labels = []
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parsed = parse_yolo_line(line)
                        if parsed:
                            labels.append(parsed)
                if labels:
                    self.prs_viewer.draw_yolo_labels(labels, w, h)
            elif polies:
                cid = self._get_prs_selected_class_id()
                if isinstance(polies[0], dict) and 'bbox' in polies[0]:
                    # Pose preview dicts: {'bbox': (bx, by, bw, bh), 'keypoints': [(kx, ky, v), ...]}
                    labels = []
                    for p in polies:
                        bx, by, bw, bh = p['bbox']
                        kpts = p.get('keypoints', [])
                        xc = (bx + bw / 2.0) / float(w)
                        yc = (by + bh / 2.0) / float(h)
                        wn = float(bw) / float(w)
                        hn = float(bh) / float(h)
                        d = [float(cid), xc, yc, wn, hn]
                        for kx, ky, kv in kpts:
                            d.extend([kx / float(w), ky / float(h), float(kv)])
                        labels.append({'type': 'pose', 'data': d})
                    self.prs_viewer.draw_yolo_labels(labels, w, h)
                elif isinstance(polies[0], (list, tuple)) and len(polies[0]) > 0 and isinstance(polies[0][0], (list, tuple)):
                    for poly_pts in polies:
                        qpoly = QPolygonF()
                        for pt in poly_pts:
                            qpoly.append(QPointF(float(pt[0]), float(pt[1])))
                        if qpoly.size() >= 3:
                            item = ResizablePolygonItem(qpoly, self.prs_viewer, "YOLO_LABEL_POLY", class_id=cid)
                            self.prs_viewer.scene.addItem(item)
                            self.prs_viewer.yolo_items.append(item)

        name = os.path.basename(filepath)
        for i in range(self.list_prs_images.count()):
            it = self.list_prs_images.item(i)
            if it.text() == name or it.text() == f"✅ {name}":
                it.setText(f"✅ {name}")
                break

    def on_prs_worker_finished(self, success_count, total_count):
        self.btn_run_prs.setEnabled(True)
        self.btn_stop_prs.setEnabled(False)
        self.btn_prs_preview.setEnabled(True)
        self.statusBar().showMessage(f"🎉 Hoàn tất: {success_count}/{total_count} ảnh thành công", 5000)
        is_pose = self.rb_prs_mode_pose.isChecked()
        mode_str = "YOLO Pose" if is_pose else "YOLO Segmentation"
        QMessageBox.information(
            self, "Hoàn tất",
            f"Đã hoàn thành dò PRS và gán nhãn {mode_str}!\n"
            f"Thành công: {success_count}/{total_count} ảnh.\n"
            f"File nhãn lưu tại: {os.path.join(self.txt_prs_output_folder.text(), 'labels')}"
        )

    def _get_image_files(self, folder):
        """Lấy danh sách file ảnh trong folder"""
        extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
        files.sort()
        return files

    # ==========================================
    # TAB 1: CROP MASTER ZONE
    # ==========================================
    def crop_current_master_zone(self):
        """Cắt ảnh Defect Zone đang chọn trên Master Image và lưu ra file"""
        if self.current_setup_img_gray is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nạp Ảnh Master trước!")
            return

        cur_row = self.list_zones.currentRow()
        zones = self._get_zones_data()
        if cur_row < 0 or cur_row >= len(zones):
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn 1 Defect Zone trong danh sách để crop!")
            return

        zone = zones[cur_row]
        if zone.get('is_ignore'):
            QMessageBox.warning(self, "Thông báo", "Vùng đang chọn là Ignore Zone. Vui lòng chọn Defect Zone để crop!")
            return
        zone_name = zone.get('name', f"Zone_{cur_row + 1}")

        # Lấy ảnh master gốc (màu nếu có, hoặc grayscale)
        master_img = None
        if hasattr(self, 'master_image_path') and self.master_image_path and os.path.exists(self.master_image_path):
            try:
                data = np.fromfile(self.master_image_path, dtype=np.uint8)
                master_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                master_img = cv2.imread(self.master_image_path)

        if master_img is None:
            master_img = self.current_setup_img_gray

        h, w = master_img.shape[:2]
        if zone.get('type') == 'polygon':
            pts = np.array(zone.get('points', []), dtype=np.int32)
            if len(pts) >= 3:
                bx, by, bw, bh = cv2.boundingRect(pts)
            else:
                QMessageBox.warning(self, "Lỗi", "Polygon không hợp lệ!")
                return
        else:
            r = zone.get('rect', [0, 0, w, h])
            bx, by, bw, bh = int(r[0]), int(r[1]), int(r[2]), int(r[3])

        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh)

        crop = master_img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            QMessageBox.warning(self, "Lỗi", "Kích thước vùng crop = 0!")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, f"Lưu ảnh crop: {zone_name}",
            f"{zone_name}.png",
            "Images (*.png *.jpg *.bmp *.tif)"
        )
        if save_path:
            try:
                ext = os.path.splitext(save_path)[1]
                is_ok, buf = cv2.imencode(ext, crop)
                if is_ok:
                    with open(save_path, 'wb') as f:
                        f.write(buf)
                    QMessageBox.information(self, "Thành công", f"Đã lưu ảnh crop {zone_name} ({x2-x1}x{y2-y1} px) vào:\n{save_path}")
                    return
            except Exception:
                pass
            cv2.imwrite(save_path, crop)
            QMessageBox.information(self, "Thành công", f"Đã lưu ảnh crop {zone_name} vào:\n{save_path}")

    # ==========================================
    # TAB 5: BATCH CROP DEFECT ZONES
    # ==========================================
    def setup_tab_crop(self):
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel bên trái: Cài đặt & Điều khiển
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(5, 5, 5, 5)
        left_lay.setSpacing(8)

        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_content = QWidget()
        lay_content = QVBoxLayout(scroll_content)
        lay_content.setSpacing(10)

        # --- Group 1: Thư mục (Input & Output) ---
        grp_folders = QGroupBox("📁 Thư mục Dữ liệu")
        lay_folders = QGridLayout()

        lay_folders.addWidget(QLabel("📂 Input Images:"), 0, 0)
        self.txt_crop_input_folder = QLineEdit()
        self.txt_crop_input_folder.setPlaceholderText("Thư mục chứa ảnh gốc cần crop...")
        lay_folders.addWidget(self.txt_crop_input_folder, 0, 1)
        btn_crop_in = QPushButton("Browse...")
        btn_crop_in.clicked.connect(self.browse_crop_input)
        lay_folders.addWidget(btn_crop_in, 0, 2)

        self.lbl_crop_file_count = QLabel("0 ảnh")
        self.lbl_crop_file_count.setStyleSheet("color: #f5a623; font-weight: bold;")
        lay_folders.addWidget(self.lbl_crop_file_count, 0, 3)

        lay_folders.addWidget(QLabel("💾 Output Crops:"), 1, 0)
        self.txt_crop_output_folder = QLineEdit()
        self.txt_crop_output_folder.setPlaceholderText("Thư mục lưu ảnh cắt (crop)...")
        lay_folders.addWidget(self.txt_crop_output_folder, 1, 1)
        btn_crop_out = QPushButton("Browse...")
        btn_crop_out.clicked.connect(self.browse_crop_output)
        lay_folders.addWidget(btn_crop_out, 1, 2)

        lay_folders.addWidget(QLabel("📄 Format:"), 2, 0)
        self.cmb_crop_format = QComboBox()
        self.cmb_crop_format.addItems([".png", ".jpg", ".bmp", ".tif"])
        lay_folders.addWidget(self.cmb_crop_format, 2, 1, 1, 2)

        lay_folders.addWidget(QLabel("🏷 Quy tắc tên:"), 3, 0)
        self.cmb_crop_naming = QComboBox()
        self.cmb_crop_naming.addItems([
            "[Tên ảnh]_[Tên Zone] (VD: img01_Zone_1.png)",
            "[Tên ảnh]_z[Số thứ tự] (VD: img01_z1.png)",
            "Tách thư mục theo Zone: output/[Tên Zone]/[Tên ảnh].png",
            "[Tên Zone]_[Tên ảnh] (VD: Zone_1_img01.png)"
        ])
        lay_folders.addWidget(self.cmb_crop_naming, 3, 1, 1, 2)

        grp_folders.setLayout(lay_folders)
        lay_content.addWidget(grp_folders)

        # --- Group 2: PRS & Căn chỉnh ---
        grp_prs = QGroupBox("🎯 Căn chỉnh PRS (Tab 1)")
        lay_prs = QVBoxLayout()

        self.chk_crop_use_prs = QCheckBox("Bật căn chỉnh PRS (tự động xoay & dịch chuyển theo Master)")
        self.chk_crop_use_prs.setChecked(True)
        self.chk_crop_use_prs.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        lay_prs.addWidget(self.chk_crop_use_prs)

        self.lbl_crop_prs_status = QLabel("Pattern 1: ❓ | Pattern 2: ❓")
        self.lbl_crop_prs_status.setStyleSheet("font-weight: bold; color: #fab387;")
        lay_prs.addWidget(self.lbl_crop_prs_status)

        lay_fallback = QHBoxLayout()
        lay_fallback.addWidget(QLabel("Khi PRS không khớp:"))
        self.cmb_crop_prs_fallback = QComboBox()
        self.cmb_crop_prs_fallback.addItems([
            "Bỏ qua ảnh (Skip)",
            "Vẫn crop theo tọa độ gốc Master (Master coords)"
        ])
        lay_fallback.addWidget(self.cmb_crop_prs_fallback)
        lay_prs.addLayout(lay_fallback)

        grp_prs.setLayout(lay_prs)
        lay_content.addWidget(grp_prs)

        # --- Group 3: Danh sách Defect Zones cần Crop ---
        grp_zones = QGroupBox("🟧 Danh sách Defect Zones cần Crop")
        lay_zones = QVBoxLayout()

        self.lbl_crop_zones_count = QLabel("Defect Zones: 0 (từ Tab 1)")
        self.lbl_crop_zones_count.setStyleSheet("color: #a6adc8;")
        lay_zones.addWidget(self.lbl_crop_zones_count)

        self.list_crop_zones = QListWidget()
        self.list_crop_zones.setMaximumHeight(130)
        lay_zones.addWidget(self.list_crop_zones)

        lay_zone_btns = QHBoxLayout()
        btn_sel_all = QPushButton("✔ Chọn tất cả")
        btn_sel_all.clicked.connect(self.select_all_crop_zones)
        btn_desel_all = QPushButton("✖ Bỏ chọn tất cả")
        btn_desel_all.clicked.connect(self.deselect_all_crop_zones)
        lay_zone_btns.addWidget(btn_sel_all)
        lay_zone_btns.addWidget(btn_desel_all)
        lay_zones.addLayout(lay_zone_btns)

        lay_margin = QHBoxLayout()
        lay_margin.addWidget(QLabel("Mở rộng vùng đệm (Margin):"))
        self.spn_crop_margin = QSpinBox()
        self.spn_crop_margin.setRange(0, 500)
        self.spn_crop_margin.setValue(0)
        self.spn_crop_margin.setSuffix(" px")
        self.spn_crop_margin.setToolTip("Mở rộng thêm vùng đệm xung quanh bounding box của defect zone")
        lay_margin.addWidget(self.spn_crop_margin)
        lay_zones.addLayout(lay_margin)

        self.chk_crop_mask_poly = QCheckBox("Masking nền đen ngoài viền đa giác (Poly Zone)")
        self.chk_crop_mask_poly.setToolTip("Nếu là vùng Polygon: Xóa đen các pixel nằm ngoài đường bao đa giác")
        lay_zones.addWidget(self.chk_crop_mask_poly)

        grp_zones.setLayout(lay_zones)
        lay_content.addWidget(grp_zones)

        # --- Group 4: Tùy chọn Resize ---
        grp_resize = QGroupBox("📐 Tùy chọn Resize kích thước")
        lay_resize = QGridLayout()

        self.chk_crop_resize = QCheckBox("Resize ảnh crop về kích thước cố định")
        self.chk_crop_resize.setChecked(False)
        lay_resize.addWidget(self.chk_crop_resize, 0, 0, 1, 4)

        lay_resize.addWidget(QLabel("Width:"), 1, 0)
        self.spn_crop_resize_w = QSpinBox()
        self.spn_crop_resize_w.setRange(16, 4096)
        self.spn_crop_resize_w.setValue(224)
        self.spn_crop_resize_w.setSuffix(" px")
        self.spn_crop_resize_w.setEnabled(False)
        lay_resize.addWidget(self.spn_crop_resize_w, 1, 1)

        lay_resize.addWidget(QLabel("Height:"), 1, 2)
        self.spn_crop_resize_h = QSpinBox()
        self.spn_crop_resize_h.setRange(16, 4096)
        self.spn_crop_resize_h.setValue(224)
        self.spn_crop_resize_h.setSuffix(" px")
        self.spn_crop_resize_h.setEnabled(False)
        lay_resize.addWidget(self.spn_crop_resize_h, 1, 3)

        self.chk_crop_resize.toggled.connect(self.spn_crop_resize_w.setEnabled)
        self.chk_crop_resize.toggled.connect(self.spn_crop_resize_h.setEnabled)

        grp_resize.setLayout(lay_resize)
        lay_content.addWidget(grp_resize)

        # --- Group 5: Nút thực thi & Tiến trình ---
        grp_actions = QGroupBox("⚡ Thực thi & Tiến trình")
        lay_actions = QVBoxLayout()

        lay_btns = QHBoxLayout()
        self.btn_crop_preview = QPushButton("👁 Thử 1 ảnh (Preview)")
        self.btn_crop_preview.setMinimumHeight(34)
        self.btn_crop_preview.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #a855f7); "
            "color: white; font-weight: bold; border-radius: 6px; font-size: 12px; }"
            "QPushButton:hover { background: #a78bfa; }"
        )
        self.btn_crop_preview.clicked.connect(self.preview_single_crop)
        lay_btns.addWidget(self.btn_crop_preview)

        self.btn_crop_run = QPushButton("🚀 BẮT ĐẦU CROP HÀNG LOẠT")
        self.btn_crop_run.setMinimumHeight(34)
        self.btn_crop_run.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4CAF50, stop:1 #66BB6A); "
            "color: white; font-weight: bold; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background: #81C784; }"
        )
        self.btn_crop_run.clicked.connect(self.toggle_run_crop)
        lay_btns.addWidget(self.btn_crop_run)

        self.btn_crop_stop = QPushButton("⏹ Dừng")
        self.btn_crop_stop.setMinimumHeight(34)
        self.btn_crop_stop.setEnabled(False)
        self.btn_crop_stop.setStyleSheet(
            "QPushButton { background: #e74c3c; color: white; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: #c0392b; }"
        )
        self.btn_crop_stop.clicked.connect(self.stop_run_crop)
        lay_btns.addWidget(self.btn_crop_stop)
        lay_actions.addLayout(lay_btns)

        self.crop_progress_bar = QProgressBar()
        self.crop_progress_bar.setValue(0)
        self.crop_progress_bar.setMinimumHeight(18)
        lay_actions.addWidget(self.crop_progress_bar)

        lay_actions.addWidget(QLabel("📋 Danh sách file ảnh:"))
        self.list_crop_images = QListWidget()
        self.list_crop_images.setMinimumHeight(140)
        self.list_crop_images.currentItemChanged.connect(self.on_crop_image_changed)
        lay_actions.addWidget(self.list_crop_images)

        lay_actions.addWidget(QLabel("📝 Log tiến trình:"))
        self.txt_crop_log = QTextEdit()
        self.txt_crop_log.setReadOnly(True)
        self.txt_crop_log.setMinimumHeight(120)
        lay_actions.addWidget(self.txt_crop_log)

        grp_actions.setLayout(lay_actions)
        lay_content.addWidget(grp_actions)

        scroll_left.setWidget(scroll_content)
        left_lay.addWidget(scroll_left)
        splitter.addWidget(left_widget)

        # Panel bên phải: Viewer ảnh toàn cảnh + kết quả Crop
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Phần trên: ImageViewer
        viewer_box = QWidget()
        v_box_lay = QVBoxLayout(viewer_box)
        v_box_lay.setContentsMargins(0, 0, 0, 0)
        lbl_v_title = QLabel("🔍 Ảnh toàn cảnh & Vị trí các vùng Crop (Defect Zones):")
        lbl_v_title.setStyleSheet("font-weight: bold; color: #f5a623; padding: 4px;")
        v_box_lay.addWidget(lbl_v_title)

        self.crop_viewer = ImageViewer()
        self.crop_viewer.current_tool = "NONE"
        v_box_lay.addWidget(self.crop_viewer)
        right_splitter.addWidget(viewer_box)

        # Phần dưới: Gallery xem các ảnh crop kết quả
        crops_box = QWidget()
        c_box_lay = QVBoxLayout(crops_box)
        c_box_lay.setContentsMargins(0, 0, 0, 0)
        lbl_c_title = QLabel("🖼 Kết quả ảnh Crop (Crops Preview):")
        lbl_c_title.setStyleSheet("font-weight: bold; color: #a6e3a1; padding: 4px;")
        c_box_lay.addWidget(lbl_c_title)

        self.scroll_crop_results = QScrollArea()
        self.scroll_crop_results.setWidgetResizable(True)
        self.scroll_crop_results.setStyleSheet("background-color: #11111b; border: 1px solid #45475a; border-radius: 6px;")
        self.scroll_crop_results.setMinimumHeight(180)

        self.crop_results_container = QWidget()
        self.crop_results_lay = QHBoxLayout(self.crop_results_container)
        self.crop_results_lay.setSpacing(12)
        self.crop_results_lay.setContentsMargins(10, 10, 10, 10)
        self.crop_results_lay.addStretch()
        self.scroll_crop_results.setWidget(self.crop_results_container)

        c_box_lay.addWidget(self.scroll_crop_results)
        right_splitter.addWidget(crops_box)

        right_splitter.setStretchFactor(0, 7)
        right_splitter.setStretchFactor(1, 3)

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        layout.addWidget(splitter)
        self.tab_crop.setLayout(layout)

    def browse_crop_input(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục Ảnh Input để Crop")
        if path:
            self.txt_crop_input_folder.setText(path)
            self.current_crop_input_folder = path
            self.refresh_crop_images_list()

    def browse_crop_output(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục Lưu Ảnh Crop")
        if path:
            self.txt_crop_output_folder.setText(path)
            self.current_crop_output_folder = path

    def refresh_crop_images_list(self):
        folder = self.txt_crop_input_folder.text().strip()
        self.list_crop_images.clear()
        if not folder or not os.path.exists(folder):
            self.lbl_crop_file_count.setText("0 ảnh")
            return
        files = self._get_image_files(folder)
        self.lbl_crop_file_count.setText(f"{len(files)} ảnh")
        for f in files:
            name = os.path.basename(f)
            self.list_crop_images.addItem(name)

    def refresh_crop_tab(self):
        """Đồng bộ trạng thái Pattern và Defect Zones từ Tab 1 vào Tab Crop"""
        self.sync_tab1_to_recipe()
        p1 = bool(self.recipe_data.get('prs1', {}).get('tpl_b64'))
        p2 = bool(self.recipe_data.get('prs2', {}).get('tpl_b64'))
        p1_txt = "✅ Đã nạp" if p1 else "❌ Chưa nạp"
        p2_txt = "✅ Đã nạp" if p2 else "❌ Chưa nạp"
        self.lbl_crop_prs_status.setText(f"Pattern 1: {p1_txt}  |  Pattern 2: {p2_txt}")

        zones = self.recipe_data.get('defect_zones', [])
        self.lbl_crop_zones_count.setText(f"Tổng số Defect Zones: {len(zones)} vùng (Tab 1)")

        from PyQt6.QtWidgets import QListWidgetItem
        old_checked = set()
        for i in range(self.list_crop_zones.count()):
            it = self.list_crop_zones.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                old_checked.add(i)

        self.list_crop_zones.clear()
        for idx, z in enumerate(zones):
            name = z.get('name', f"Zone_{idx + 1}")
            z_type = z.get('type', 'rect')
            if z_type == 'polygon':
                pts_cnt = len(z.get('points', []))
                text = f"{name} [Polygon: {pts_cnt} pts]"
            else:
                rect = z.get('rect', [0, 0, 0, 0])
                text = f"{name} [Rect: {rect[2]}x{rect[3]}]"

            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if not old_checked or idx in old_checked:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.list_crop_zones.addItem(item)

    def select_all_crop_zones(self):
        for i in range(self.list_crop_zones.count()):
            self.list_crop_zones.item(i).setCheckState(Qt.CheckState.Checked)

    def deselect_all_crop_zones(self):
        for i in range(self.list_crop_zones.count()):
            self.list_crop_zones.item(i).setCheckState(Qt.CheckState.Unchecked)

    def on_crop_image_changed(self, current, previous):
        if current is not None:
            self.on_crop_image_selected(current)

    def on_crop_image_selected(self, item):
        name = item.text().replace("✅ ", "")
        folder = self.txt_crop_input_folder.text().strip()
        img_path = os.path.join(folder, name)
        if os.path.exists(img_path):
            self.preview_single_crop(img_path)

    def _get_batch_crop_config(self):
        self.sync_tab1_to_recipe()
        folder = self.txt_crop_input_folder.text().strip()
        out_folder = self.txt_crop_output_folder.text().strip()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục Ảnh Input hợp lệ!")
            return None
        if not out_folder:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục Output Lưu Ảnh Crop!")
            return None

        files = self._get_image_files(folder)
        if not files:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy file ảnh nào trong thư mục Input!")
            return None

        selected_indices = []
        for i in range(self.list_crop_zones.count()):
            if self.list_crop_zones.item(i).checkState() == Qt.CheckState.Checked:
                selected_indices.append(i)

        if not selected_indices:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ít nhất 1 Defect Zone để crop!")
            return None

        naming_keys = ["img_zone", "img_idx", "zone_folder", "zone_img"]
        naming_idx = self.cmb_crop_naming.currentIndex()
        naming_mode = naming_keys[naming_idx] if 0 <= naming_idx < len(naming_keys) else "img_zone"

        prs_fallback = "skip" if self.cmb_crop_prs_fallback.currentIndex() == 0 else "master_coord"

        cfg = {
            'recipe': self.recipe_data,
            'input_files': files,
            'output_folder': out_folder,
            'selected_zone_indices': selected_indices,
            'use_prs': self.chk_crop_use_prs.isChecked(),
            'prs_fallback': prs_fallback,
            'margin': self.spn_crop_margin.value(),
            'image_format': self.cmb_crop_format.currentText(),
            'naming_mode': naming_mode,
            'mask_polygon': self.chk_crop_mask_poly.isChecked(),
            'resize_enabled': self.chk_crop_resize.isChecked(),
            'resize_w': self.spn_crop_resize_w.value(),
            'resize_h': self.spn_crop_resize_h.value()
        }
        return cfg

    def preview_single_crop(self, img_path=None):
        """Thử crop trên 1 ảnh và hiển thị kết quả lên viewer + panel cards"""
        try:
            self.sync_tab1_to_recipe()
            folder = self.txt_crop_input_folder.text().strip()
            if not folder or not os.path.exists(folder):
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục Ảnh Input trước!")
                return

            if not img_path:
                curr_item = self.list_crop_images.currentItem()
                if curr_item:
                    name = curr_item.text().replace("✅ ", "")
                    img_path = os.path.join(folder, name)
                else:
                    files = self._get_image_files(folder)
                    if not files:
                        QMessageBox.warning(self, "Lỗi", "Không có file ảnh nào trong thư mục Input!")
                        return
                    img_path = files[0]

            out_folder = self.txt_crop_output_folder.text().strip()
            if not out_folder:
                out_folder = os.path.join(folder, "crops_preview")

            selected_indices = []
            for i in range(self.list_crop_zones.count()):
                if self.list_crop_zones.item(i).checkState() == Qt.CheckState.Checked:
                    selected_indices.append(i)

            if not selected_indices:
                selected_indices = list(range(self.list_crop_zones.count()))

            naming_keys = ["img_zone", "img_idx", "zone_folder", "zone_img"]
            naming_idx = self.cmb_crop_naming.currentIndex()
            naming_mode = naming_keys[naming_idx] if 0 <= naming_idx < len(naming_keys) else "img_zone"
            prs_fallback = "skip" if self.cmb_crop_prs_fallback.currentIndex() == 0 else "master_coord"

            cfg = {
                'recipe': self.recipe_data,
                'input_files': [img_path],
                'output_folder': out_folder,
                'selected_zone_indices': selected_indices,
                'use_prs': self.chk_crop_use_prs.isChecked(),
                'prs_fallback': prs_fallback,
                'margin': self.spn_crop_margin.value(),
                'image_format': self.cmb_crop_format.currentText(),
                'naming_mode': naming_mode,
                'mask_polygon': self.chk_crop_mask_poly.isChecked(),
                'resize_enabled': self.chk_crop_resize.isChecked(),
                'resize_w': self.spn_crop_resize_w.value(),
                'resize_h': self.spn_crop_resize_h.value()
            }

            worker = BatchCropWorkerThread(cfg)
            success, aligned_img, crops = worker.process_single_image(img_path)
            if not success or aligned_img is None:
                QMessageBox.warning(self, "Thông báo", f"Không crop được ảnh:\n{os.path.basename(img_path)}")
                return

            self.crop_viewer.load_image_from_array(aligned_img)
            self.draw_crop_zones_overlay(crops)
            self.update_crop_results_display(crops)

            self.txt_crop_log.append(f"👁 Preview {os.path.basename(img_path)}: Đã crop {len(crops)} vùng defect zone.")
            self.statusBar().showMessage(f"👁 Preview crop thành công: {len(crops)} vùng", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Preview", f"Đã xảy ra lỗi khi tạo preview:\n{e}")
            if hasattr(self, 'txt_crop_log'):
                self.txt_crop_log.append(f"❌ Lỗi Preview: {e}")

    def toggle_run_crop(self):
        cfg = self._get_batch_crop_config()
        if not cfg:
            return

        self.btn_crop_run.setEnabled(False)
        self.btn_crop_stop.setEnabled(True)
        self.btn_crop_preview.setEnabled(False)
        self.crop_progress_bar.setValue(0)
        self.txt_crop_log.clear()

        self.crop_worker = BatchCropWorkerThread(cfg)
        self.crop_worker.log_signal.connect(self.on_crop_worker_log)
        self.crop_worker.progress_signal.connect(self.on_crop_worker_progress)
        self.crop_worker.preview_signal.connect(self.on_crop_worker_preview)
        self.crop_worker.finished_signal.connect(self.on_crop_worker_finished)
        self.crop_worker.start()

    def stop_run_crop(self):
        if self.crop_worker and self.crop_worker.isRunning():
            self.crop_worker.stop()

    def on_crop_worker_log(self, msg):
        self.txt_crop_log.append(msg)
        self.txt_crop_log.moveCursor(QTextCursor.MoveOperation.End)

    def on_crop_worker_progress(self, current, total):
        pct = int((current / total) * 100) if total > 0 else 0
        self.crop_progress_bar.setValue(pct)

    def on_crop_worker_preview(self, aligned_img, filepath, crops):
        if aligned_img is not None:
            self.crop_viewer.load_image_from_array(aligned_img)
            self.draw_crop_zones_overlay(crops)
            self.update_crop_results_display(crops)

        name = os.path.basename(filepath)
        for i in range(self.list_crop_images.count()):
            item = self.list_crop_images.item(i)
            if item.text().replace("✅ ", "") == name:
                item.setText(f"✅ {name}")
                break

    def on_crop_worker_finished(self, success_count, total_count, total_crops):
        self.btn_crop_run.setEnabled(True)
        self.btn_crop_stop.setEnabled(False)
        self.btn_crop_preview.setEnabled(True)
        self.statusBar().showMessage(f"Hoàn tất Crop: {success_count}/{total_count} ảnh, {total_crops} ảnh crop", 5000)
        QMessageBox.information(
            self, "Hoàn tất",
            f"🎉 Đã hoàn tất Batch Crop Defect Zones!\n\n"
            f"• Số ảnh thành công: {success_count} / {total_count}\n"
            f"• Tổng số file crop đã lưu: {total_crops}\n"
            f"• Thư mục output:\n{self.txt_crop_output_folder.text()}"
        )

    def draw_crop_zones_overlay(self, crops):
        """Vẽ các khung viền chữ nhật màu cam/vàng thể hiện các vùng Defect Zone đã crop trên viewer"""
        for it in list(self.crop_viewer.zone_items):
            try:
                self.crop_viewer.scene.removeItem(it)
            except Exception:
                pass
        self.crop_viewer.zone_items.clear()

        from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem
        from PyQt6.QtGui import QPen, QBrush, QColor, QFont
        from PyQt6.QtCore import QRectF, QPointF

        for zone_name, crop_img, out_path, rect_coords, poly_pts in crops:
            x, y, w, h = rect_coords
            if poly_pts is not None and len(poly_pts) >= 3:
                qpoly = QPolygonF()
                for pt in poly_pts:
                    qpoly.append(QPointF(float(pt[0]), float(pt[1])))
                item = QGraphicsPolygonItem(qpoly)
                item.setPen(QPen(QColor(245, 166, 35), 2))
                item.setBrush(QBrush(QColor(245, 166, 35, 40)))
                self.crop_viewer.scene.addItem(item)
                self.crop_viewer.zone_items.append(item)
            else:
                rect_item = QGraphicsRectItem(QRectF(float(x), float(y), float(w), float(h)))
                rect_item.setPen(QPen(QColor(245, 166, 35), 2))
                rect_item.setBrush(QBrush(QColor(245, 166, 35, 40)))
                self.crop_viewer.scene.addItem(rect_item)
                self.crop_viewer.zone_items.append(rect_item)

            txt_item = QGraphicsSimpleTextItem(zone_name)
            txt_item.setPos(float(x + 2), max(0.0, float(y - 18)))
            txt_item.setBrush(QBrush(QColor(255, 255, 255)))
            font = QFont("Arial", 9, QFont.Weight.Bold)
            txt_item.setFont(font)
            self.crop_viewer.scene.addItem(txt_item)
            self.crop_viewer.zone_items.append(txt_item)

    def update_crop_results_display(self, crops):
        """Xóa và nạp danh sách card ảnh crop vào panel preview"""
        while self.crop_results_lay.count():
            item = self.crop_results_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not crops:
            lbl_empty = QLabel("Chưa có ảnh crop nào được hiển thị.")
            lbl_empty.setStyleSheet("color: #6c7086; font-style: italic; padding: 20px;")
            self.crop_results_lay.addWidget(lbl_empty)
            self.crop_results_lay.addStretch()
            return

        for zone_name, crop_img, out_path, rect_coords, poly_pts in crops:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #1e1e2e;
                    border: 1px solid #585b70;
                    border-radius: 8px;
                    padding: 6px;
                }
                QFrame:hover {
                    border-color: #f5a623;
                }
            """)
            card_lay = QVBoxLayout(card)
            card_lay.setSpacing(4)
            card_lay.setContentsMargins(6, 6, 6, 6)

            lbl_thumb = QLabel()
            lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_thumb.setStyleSheet("background-color: #000000; border-radius: 4px;")
            lbl_thumb.setFixedSize(140, 120)

            if crop_img is not None and crop_img.size > 0:
                h_c, w_c = crop_img.shape[:2]
                if len(crop_img.shape) == 2:
                    rgb = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                pix = numpy_to_qpixmap(rgb)
                scaled_pix = pix.scaled(136, 116, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                lbl_thumb.setPixmap(scaled_pix)
                dim_text = f"{w_c} × {h_c} px"
            else:
                dim_text = "N/A"

            card_lay.addWidget(lbl_thumb)

            lbl_name = QLabel(zone_name)
            lbl_name.setStyleSheet("font-weight: bold; color: #f5a623; font-size: 11px;")
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_lay.addWidget(lbl_name)

            lbl_dim = QLabel(dim_text)
            lbl_dim.setStyleSheet("color: #a6adc8; font-size: 10px;")
            lbl_dim.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_lay.addWidget(lbl_dim)

            if out_path and os.path.exists(out_path):
                btn_open = QPushButton("📂 Mở")
                btn_open.setMaximumHeight(22)
                btn_open.setStyleSheet("font-size: 10px; padding: 2px 6px;")
                btn_open.clicked.connect(lambda _, p=out_path: os.system(f'explorer /select,"{os.path.normpath(p)}"'))
                card_lay.addWidget(btn_open)

            self.crop_results_lay.addWidget(card)

        self.crop_results_lay.addStretch()

    # ==========================================
    # TAB 6: DATASET MANAGER (EXPLORER & BUILDER)
    # ==========================================
    def setup_tab_dataset(self):
        """Khởi tạo Tab 6 với 2 sub-tabs: Dataset Explorer & Dataset Builder"""
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        self.subtabs_dataset = QTabWidget()
        self.subtab_explorer = QWidget()
        self.setup_subtab_explorer()
        self.subtabs_dataset.addTab(self.subtab_explorer, "🔍 Dataset Explorer (Khảo sát Dataset gốc)")

        self.subtab_builder = QWidget()
        self.setup_subtab_builder()
        self.subtabs_dataset.addTab(self.subtab_builder, "🏗 Dataset Builder (Tạo Dataset mới)")

        self.subtabs_dataset.currentChanged.connect(self.on_dataset_subtab_changed)
        layout.addWidget(self.subtabs_dataset)
        self.tab_dataset.setLayout(layout)

    def on_dataset_subtab_changed(self, index):
        if index == 1:
            self.refresh_builder_sources_ui()

    def refresh_dataset_tab(self):
        """Đồng bộ khi người dùng click vào Tab 6"""
        if hasattr(self, 'table_builder_src'):
            self.refresh_builder_sources_ui()

    # ------------------------------------------
    # SUBTAB 1: DATASET EXPLORER
    # ------------------------------------------
    def setup_subtab_explorer(self):
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === PANEL BÊN TRÁI ===
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(6)

        # 1. Thư mục dataset nguồn
        grp_sources = QGroupBox("📁 Thư mục Dataset YOLO Gốc")
        lay_src = QVBoxLayout()
        self.list_dataset_sources = QListWidget()
        self.list_dataset_sources.setMaximumHeight(90)
        lay_src.addWidget(self.list_dataset_sources)

        self.lbl_selected_ds_hint = QLabel("👉 Thư mục đang chọn: (Chưa chọn)")
        self.lbl_selected_ds_hint.setStyleSheet("color: #89b4fa; font-size: 11px;")
        lay_src.addWidget(self.lbl_selected_ds_hint)
        self.list_dataset_sources.currentItemChanged.connect(self.on_dataset_source_item_changed)

        lay_src_btns = QHBoxLayout()
        btn_add_ds = QPushButton("➕ Thêm Thư mục")
        btn_add_ds.clicked.connect(self.add_dataset_source)
        btn_del_ds = QPushButton("❌ Xóa")
        btn_del_ds.clicked.connect(self.del_dataset_source)
        btn_scan_all = QPushButton("🔄 Quét (Scan) Tất cả")
        btn_scan_all.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1e3c72, stop:1 #2a5298); "
            "color: white; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background: #3b6cb7; }"
        )
        btn_scan_all.clicked.connect(self.scan_all_datasets)
        lay_src_btns.addWidget(btn_add_ds)
        lay_src_btns.addWidget(btn_del_ds)
        lay_src_btns.addWidget(btn_scan_all)
        lay_src.addLayout(lay_src_btns)
        grp_sources.setLayout(lay_src)
        left_lay.addWidget(grp_sources)

        # 2. Thống kê & Phân bố Class
        grp_stats = QGroupBox("📊 Thống kê & Phân bố Class")
        lay_stats = QVBoxLayout()

        lay_summary = QHBoxLayout()
        self.lbl_stat_images = QLabel("📷 0 ảnh")
        self.lbl_stat_images.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        self.lbl_stat_objects = QLabel("🏷 0 vật thể")
        self.lbl_stat_objects.setStyleSheet("font-weight: bold; color: #a6e3a1; font-size: 12px;")
        self.lbl_stat_bg = QLabel("⚪ 0 ảnh background")
        self.lbl_stat_bg.setStyleSheet("font-weight: bold; color: #fab387; font-size: 12px;")
        lay_summary.addWidget(self.lbl_stat_images)
        lay_summary.addWidget(self.lbl_stat_objects)
        lay_summary.addWidget(self.lbl_stat_bg)
        lay_stats.addLayout(lay_summary)

        self.table_class_dist = QTableWidget()
        self.table_class_dist.setColumnCount(6)
        self.table_class_dist.setHorizontalHeaderLabels(["Chọn", "ID", "Tên Class", "Số vật thể", "Tỉ lệ %", "Số ảnh chứa"])
        self.table_class_dist.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_class_dist.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_class_dist.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_class_dist.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_class_dist.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_class_dist.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table_class_dist.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_class_dist.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_class_dist.setMaximumHeight(160)
        self.table_class_dist.setStyleSheet("""
            QTableWidget { background-color: #181825; gridline-color: #313244; }
            QHeaderView::section { background-color: #313244; color: #cdd6f4; font-weight: bold; }
        """)
        lay_stats.addWidget(self.table_class_dist)

        # Hàng nút thao tác: Chọn / Bỏ chọn & Xóa Class đã chọn
        lay_class_actions = QHBoxLayout()
        btn_sel_all_cls = QPushButton("☑ Chọn tất cả")
        btn_sel_all_cls.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_sel_all_cls.clicked.connect(self.select_all_classes_in_table)
        btn_desel_all_cls = QPushButton("☐ Bỏ chọn")
        btn_desel_all_cls.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_desel_all_cls.clicked.connect(self.deselect_all_classes_in_table)
        lay_class_actions.addWidget(btn_sel_all_cls)
        lay_class_actions.addWidget(btn_desel_all_cls)

        self.btn_bulk_del_classes = QPushButton("🗑️ Xóa Class đã chọn (Thư mục đang chọn)")
        self.btn_bulk_del_classes.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                          stop:0 #d63031, stop:1 #e17055);
                          color: white; font-weight: bold; font-size: 11px;
                          border: none; border-radius: 5px; padding: 4px 12px; min-height: 22px; }
            QPushButton:hover { background: #ff7675; }
        """)
        self.btn_bulk_del_classes.clicked.connect(self.bulk_delete_selected_classes)
        lay_class_actions.addWidget(self.btn_bulk_del_classes, stretch=1)
        lay_stats.addLayout(lay_class_actions)

        grp_stats.setLayout(lay_stats)
        left_lay.addWidget(grp_stats)

        # 3. Danh sách ảnh & Bộ lọc
        grp_imgs = QGroupBox("📋 Danh sách Ảnh & Bộ lọc")
        lay_imgs = QVBoxLayout()

        lay_filters = QHBoxLayout()
        self.txt_filter_img_name = QLineEdit()
        self.txt_filter_img_name.setPlaceholderText("🔍 Tìm tên file...")
        self.txt_filter_img_name.textChanged.connect(self.filter_explorer_images)
        lay_filters.addWidget(self.txt_filter_img_name)

        self.cmb_filter_class = QComboBox()
        self.cmb_filter_class.addItem("Tất cả ảnh")
        self.cmb_filter_class.currentIndexChanged.connect(self.filter_explorer_images)
        lay_filters.addWidget(self.cmb_filter_class)
        lay_imgs.addLayout(lay_filters)

        self.list_explorer_images = QListWidget()
        self.list_explorer_images.currentItemChanged.connect(self.on_explorer_image_changed)
        lay_imgs.addWidget(self.list_explorer_images)

        self.lbl_explorer_img_info = QLabel("Chưa chọn ảnh")
        self.lbl_explorer_img_info.setStyleSheet("color: #a6adc8; font-size: 11px;")
        lay_imgs.addWidget(self.lbl_explorer_img_info)

        grp_imgs.setLayout(lay_imgs)
        left_lay.addWidget(grp_imgs)

        splitter.addWidget(left_widget)

        # === PANEL BÊN PHẢI: Image Viewer + Bảng đối tượng ===
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        top_v = QWidget()
        top_v_lay = QVBoxLayout(top_v)
        top_v_lay.setContentsMargins(0, 0, 0, 0)
        lbl_v_title = QLabel("🔍 Trình xem ảnh & Bounding Box (YOLO Viewer):")
        lbl_v_title.setStyleSheet("font-weight: bold; color: #f5a623; padding: 2px;")
        top_v_lay.addWidget(lbl_v_title)

        self.explorer_viewer = ImageViewer()
        self.explorer_viewer.current_tool = "NONE"
        top_v_lay.addWidget(self.explorer_viewer)
        right_splitter.addWidget(top_v)

        bot_v = QWidget()
        bot_v_lay = QVBoxLayout(bot_v)
        bot_v_lay.setContentsMargins(0, 0, 0, 0)
        lbl_obj_title = QLabel("🏷 Danh sách vật thể trong ảnh đang chọn:")
        lbl_obj_title.setStyleSheet("font-weight: bold; color: #a6e3a1; padding: 2px;")
        bot_v_lay.addWidget(lbl_obj_title)

        self.table_image_objects = QTableWidget()
        self.table_image_objects.setColumnCount(4)
        self.table_image_objects.setHorizontalHeaderLabels(["#", "Class ID", "Tên Class", "Tọa độ BBox (x_c, y_c, w, h)"])
        self.table_image_objects.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_image_objects.horizontalHeader().setStretchLastSection(True)
        self.table_image_objects.setMaximumHeight(130)
        self.table_image_objects.setStyleSheet("""
            QTableWidget { background-color: #181825; gridline-color: #313244; }
            QHeaderView::section { background-color: #313244; color: #cdd6f4; font-weight: bold; }
        """)
        bot_v_lay.addWidget(self.table_image_objects)
        right_splitter.addWidget(bot_v)

        right_splitter.setStretchFactor(0, 8)
        right_splitter.setStretchFactor(1, 2)

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        layout.addWidget(splitter)
        self.subtab_explorer.setLayout(layout)

    def add_dataset_source(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục Dataset YOLO gốc")
        if path:
            norm_p = os.path.normpath(path)
            if norm_p not in self.dataset_source_paths:
                self.dataset_source_paths.append(norm_p)
                self.list_dataset_sources.addItem(norm_p)
                self.scan_all_datasets()
            else:
                QMessageBox.information(self, "Thông báo", "Thư mục này đã có trong danh sách nguồn!")

    def del_dataset_source(self):
        row = self.list_dataset_sources.currentRow()
        if row >= 0:
            self.dataset_source_paths.pop(row)
            self.list_dataset_sources.takeItem(row)
            if self.dataset_source_paths:
                self.scan_all_datasets()
            else:
                self.scanned_dataset_data = {}
                self.update_explorer_statistics_ui()
                self.filter_explorer_images()
                self.refresh_builder_sources_ui()

    def on_dataset_source_item_changed(self, current, previous):
        if current:
            p = current.text().strip()
            self.lbl_selected_ds_hint.setText(f"👉 Thư mục đang chọn: <b>{os.path.basename(p)}</b>")
            self.lbl_selected_ds_hint.setToolTip(p)
        else:
            self.lbl_selected_ds_hint.setText("👉 Thư mục đang chọn: (Chưa chọn)")
            self.lbl_selected_ds_hint.setToolTip("")

    def select_all_classes_in_table(self):
        for r in range(self.table_class_dist.rowCount()):
            item = self.table_class_dist.item(r, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

    def deselect_all_classes_in_table(self):
        for r in range(self.table_class_dist.rowCount()):
            item = self.table_class_dist.item(r, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_class_ids_from_table(self):
        cids = []
        for r in range(self.table_class_dist.rowCount()):
            chk_item = self.table_class_dist.item(r, 0)
            if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                id_item = self.table_class_dist.item(r, 1)
                if id_item:
                    try:
                        cids.append(int(id_item.text()))
                    except ValueError:
                        pass
        if not cids:
            sel_rows = set(index.row() for index in self.table_class_dist.selectedIndexes())
            for r in sel_rows:
                id_item = self.table_class_dist.item(r, 1)
                if id_item:
                    try:
                        cids.append(int(id_item.text()))
                    except ValueError:
                        pass
        return sorted(list(set(cids)))

    def bulk_delete_selected_classes(self):
        """Xóa hàng loạt các class được chọn - CHỈ áp dụng đối với thư mục dataset được chọn"""
        # 1. Kiểm tra thư mục dataset được chọn
        curr_item = self.list_dataset_sources.currentItem()
        target_dir = curr_item.text().strip() if curr_item else ""

        if not target_dir:
            if self.list_dataset_sources.count() == 1:
                target_dir = self.list_dataset_sources.item(0).text().strip()
                self.list_dataset_sources.setCurrentRow(0)
            elif self.list_dataset_sources.count() > 1:
                QMessageBox.warning(
                    self, "Chưa chọn thư mục",
                    "⚠️ Vui lòng nhấp chọn 1 thư mục Dataset trong danh sách 'Thư mục Dataset YOLO Gốc' ở trên để tiến hành xóa!"
                )
                return
            else:
                QMessageBox.warning(
                    self, "Chưa có Dataset",
                    "Chưa có thư mục dataset nào trong danh sách nguồn!"
                )
                return

        if not os.path.exists(target_dir):
            QMessageBox.warning(self, "Lỗi", f"Thư mục dataset không tồn tại:\n{target_dir}")
            return

        # 2. Kiểm tra các class được chọn
        selected_cids = self._get_selected_class_ids_from_table()
        if not selected_cids:
            QMessageBox.warning(
                self, "Chưa chọn Class",
                "⚠️ Vui lòng tích chọn ít nhất 1 class trong bảng 'Thống kê & Phân bố Class' để xóa!"
            )
            return

        # Lấy thông tin chi tiết các class được chọn trong dataset này
        data = self.scanned_dataset_data
        all_names = data.get('all_class_names', {})

        target_norm = os.path.normpath(target_dir)
        matched_ds = None
        for ds in data.get('datasets', []):
            if os.path.normpath(ds.get('path', '')) == target_norm:
                matched_ds = ds
                break

        ds_counts = matched_ds.get('class_counts', {}) if matched_ds else {}
        total_objs_to_delete = sum(ds_counts.get(cid, 0) for cid in selected_cids)

        # 3. HỘP THOẠI XÁC NHẬN BẢO MẬT (Hỏi lại trước khi tiến hành xóa để tránh bấm nhầm)
        dialog = QDialog(self)
        dialog.setWindowTitle("⚠️ Xác nhận Xóa Class Hàng Loạt")
        dialog.setMinimumWidth(540)
        lay = QVBoxLayout(dialog)
        lay.setSpacing(10)

        lbl_warn = QLabel("⚠️ CẢNH BÁO: BẠN ĐANG CHUẨN BỊ XÓA CLASS HÀNG LOẠT KHỎI DATASET!")
        lbl_warn.setStyleSheet("color: #ff6b6b; font-weight: bold; font-size: 13px;")
        lay.addWidget(lbl_warn)

        grp_scope = QGroupBox("📁 Thư mục áp dụng xóa")
        lay_sc = QVBoxLayout(grp_scope)
        lbl_scope = QLabel(f"<b>Thư mục được chọn:</b><br><span style='color: #89b4fa;'>{target_dir}</span>")
        lbl_scope.setWordWrap(True)
        lay_sc.addWidget(lbl_scope)
        lay.addWidget(grp_scope)

        grp_cls = QGroupBox("🏷 Các Class sẽ bị xóa bỏ")
        lay_cls = QVBoxLayout(grp_cls)
        cls_summary_lines = []
        for cid in selected_cids:
            cname = all_names.get(cid, f"class_{cid}")
            cnt = ds_counts.get(cid, 0)
            cls_summary_lines.append(f"• <b>Class {cid} ({cname})</b>: {cnt:,} vật thể trong thư mục này")
        lbl_cls_list = QLabel("<br>".join(cls_summary_lines))
        lay_cls.addWidget(lbl_cls_list)

        lbl_tot = QLabel(f"➡️ <b>Tổng cộng sẽ xóa: <span style='color: #ff6b6b;'>{total_objs_to_delete:,} vật thể</span></b>")
        lbl_tot.setStyleSheet("font-size: 12px; margin-top: 4px;")
        lay_cls.addWidget(lbl_tot)
        lay.addWidget(grp_cls)

        grp_opts = QGroupBox("⚙️ Tùy chọn xử lý")
        lay_opt = QVBoxLayout(grp_opts)

        chk_reindex = QCheckBox("🔄 Đánh số lại ID các class còn lại liên tục (0, 1, 2...) và cập nhật data.yaml")
        chk_reindex.setChecked(True)
        chk_reindex.setToolTip("Tránh lỗ hổng ID sau khi xóa class (ví dụ xóa class 1 thì class 2 sẽ chuyển thành class 1)")
        lay_opt.addWidget(chk_reindex)

        chk_del_empty_imgs = QCheckBox("🗑️ Xóa luôn file ảnh nếu ảnh đó không còn bất kỳ vật thể nào khác")
        chk_del_empty_imgs.setChecked(False)
        chk_del_empty_imgs.setToolTip("Nếu không chọn: Ảnh không còn vật thể sẽ được giữ lại làm ảnh Background (chuẩn YOLO)")
        lay_opt.addWidget(chk_del_empty_imgs)

        chk_backup = QCheckBox("💾 Tạo bản sao lưu file nhãn (.bak) trước khi chỉnh sửa")
        chk_backup.setChecked(True)
        lay_opt.addWidget(chk_backup)

        lay.addWidget(grp_opts)

        lay_btns = QHBoxLayout()
        btn_cancel = QPushButton("❌ Hủy bỏ (Không xóa)")
        btn_cancel.setMinimumHeight(36)
        btn_cancel.clicked.connect(dialog.reject)

        btn_confirm = QPushButton("🗑️ XÁC NHẬN XÓA")
        btn_confirm.setMinimumHeight(36)
        btn_confirm.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #d63031, stop:1 #e17055);
                          color: white; font-weight: bold; font-size: 12px; border-radius: 6px; padding: 6px 16px; }
            QPushButton:hover { background: #ff7675; }
        """)
        btn_confirm.clicked.connect(dialog.accept)

        lay_btns.addWidget(btn_cancel)
        lay_btns.addWidget(btn_confirm)
        lay.addLayout(lay_btns)

        btn_cancel.setDefault(True)
        btn_cancel.setFocus()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Đã hủy thao tác xóa class.", 3000)
            return

        # 4. TIẾN HÀNH XÓA (Chỉ đối với target_dir)
        do_reindex = chk_reindex.isChecked()
        do_del_imgs = chk_del_empty_imgs.isChecked()
        do_backup = chk_backup.isChecked()

        remaining_cids = sorted([cid for cid in ds_counts.keys() if cid not in selected_cids])
        reindex_map = {old_cid: new_cid for new_cid, old_cid in enumerate(remaining_cids)} if do_reindex else {}

        self.statusBar().showMessage(f"⏳ Đang xóa class khỏi {os.path.basename(target_dir)}...", 5000)

        import shutil
        modified_files_count = 0
        deleted_objects_count = 0
        deleted_images_count = 0

        label_files_to_process = set()
        img_map_for_labels = {}

        if matched_ds and matched_ds.get('images'):
            for rec in matched_ds['images']:
                lp = rec.get('label_path')
                if lp and os.path.isfile(lp):
                    label_files_to_process.add(lp)
                    img_map_for_labels[lp] = rec.get('img_path')
        else:
            for root, _, files in os.walk(target_dir):
                for f in files:
                    if f.lower().endswith('.txt') and not f.lower().endswith('.bak'):
                        lp = os.path.join(root, f)
                        label_files_to_process.add(lp)

        for lp in label_files_to_process:
            if not os.path.isfile(lp):
                continue

            try:
                with open(lp, 'r', encoding='utf-8', errors='ignore') as lf:
                    lines = lf.readlines()

                new_lines = []
                file_changed = False

                for line in lines:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    parts = line_str.split()
                    try:
                        cid = int(parts[0])
                        if cid in selected_cids:
                            file_changed = True
                            deleted_objects_count += 1
                            continue
                        
                        if do_reindex and cid in reindex_map:
                            new_cid = reindex_map[cid]
                            if new_cid != cid:
                                parts[0] = str(new_cid)
                                line_str = " ".join(parts)
                                file_changed = True
                    except ValueError:
                        pass
                    
                    new_lines.append(line_str)

                if file_changed:
                    if do_backup:
                        bak_path = lp + ".bak"
                        if not os.path.exists(bak_path):
                            shutil.copy2(lp, bak_path)

                    if not new_lines and do_del_imgs:
                        os.remove(lp)
                        img_p = img_map_for_labels.get(lp)
                        if img_p and os.path.isfile(img_p):
                            os.remove(img_p)
                            deleted_images_count += 1
                    else:
                        with open(lp, 'w', encoding='utf-8') as lf:
                            if new_lines:
                                lf.write("\n".join(new_lines) + "\n")

                    modified_files_count += 1

            except Exception as e:
                print(f"Lỗi khi xử lý file {lp}: {e}")

        # Cập nhật data.yaml trong target_dir nếu có
        yaml_path = os.path.join(target_dir, "data.yaml")
        if not os.path.isfile(yaml_path):
            yaml_path = os.path.join(target_dir, "dataset.yaml")

        if os.path.isfile(yaml_path):
            try:
                import yaml
                with open(yaml_path, 'r', encoding='utf-8') as yf:
                    yaml_data = yaml.safe_load(yf)

                if isinstance(yaml_data, dict) and 'names' in yaml_data:
                    raw_names = yaml_data['names']
                    if isinstance(raw_names, list):
                        if do_reindex:
                            new_names_list = [raw_names[cid] for cid in remaining_cids if cid < len(raw_names)]
                        else:
                            new_names_list = [name for cid, name in enumerate(raw_names) if cid not in selected_cids]
                        yaml_data['names'] = new_names_list
                        yaml_data['nc'] = len(new_names_list)
                    elif isinstance(raw_names, dict):
                        if do_reindex:
                            new_names_dict = {new_cid: raw_names.get(old_cid, raw_names.get(str(old_cid), f"class_{old_cid}"))
                                              for new_cid, old_cid in enumerate(remaining_cids)}
                        else:
                            new_names_dict = {k: v for k, v in raw_names.items() if int(k) not in selected_cids}
                        yaml_data['names'] = new_names_dict
                        yaml_data['nc'] = len(new_names_dict)

                    with open(yaml_path, 'w', encoding='utf-8') as yf:
                        yaml.dump(yaml_data, yf, default_flow_style=False, allow_unicode=True)
            except Exception as e:
                print(f"Lỗi cập nhật data.yaml: {e}")

        # 5. Thông báo kết quả và quét lại dataset
        msg = (
            f"✅ ĐÃ HOÀN TẤT XÓA CLASS!\n\n"
            f"📁 Thư mục áp dụng: {os.path.basename(target_dir)}\n"
            f"🏷 Classes đã xóa: {', '.join(f'Class {c}' for c in selected_cids)}\n"
            f"✂️ Số vật thể đã xóa: {deleted_objects_count:,}\n"
            f"📝 Số file nhãn đã cập nhật: {modified_files_count:,}"
        )
        if do_del_imgs and deleted_images_count > 0:
            msg += f"\n🗑️ Số ảnh rỗng đã xóa: {deleted_images_count:,}"
        if do_reindex:
            msg += f"\n🔄 Đã đánh số lại ID liên tục (0..{len(remaining_cids)-1}) và cập nhật file data.yaml."

        QMessageBox.information(self, "Hoàn tất xóa Class", msg)
        self.scan_all_datasets()

    def scan_all_datasets(self):
        if not self.dataset_source_paths:
            QMessageBox.warning(self, "Lỗi", "Vui lòng thêm ít nhất 1 thư mục dataset để quét!")
            return

        self.statusBar().showMessage("🔍 Đang quét dataset...", 3000)
        self.dataset_scan_worker = DatasetScanWorkerThread(self.dataset_source_paths)
        self.dataset_scan_worker.finished_signal.connect(self.on_dataset_scan_finished)
        self.dataset_scan_worker.start()

    def on_dataset_scan_finished(self, result):
        self.scanned_dataset_data = result
        self.update_explorer_statistics_ui()
        self.filter_explorer_images()
        self.refresh_builder_sources_ui()
        self.statusBar().showMessage(f"Quét hoàn tất: {result['total_images']} ảnh, {result['total_objects']} vật thể", 4000)

    def update_explorer_statistics_ui(self):
        data = self.scanned_dataset_data
        tot_img = data.get('total_images', 0)
        tot_obj = data.get('total_objects', 0)
        tot_bg = data.get('empty_images_count', 0)

        self.lbl_stat_images.setText(f"📷 {tot_img:,} ảnh")
        self.lbl_stat_objects.setText(f"🏷 {tot_obj:,} vật thể")
        self.lbl_stat_bg.setText(f"⚪ {tot_bg:,} background")

        all_counts = data.get('all_class_counts', {})
        all_img_counts = data.get('all_class_images_counts', {})
        all_names = data.get('all_class_names', {})

        self.table_class_dist.setRowCount(0)
        self.cmb_filter_class.blockSignals(True)
        self.cmb_filter_class.clear()
        self.cmb_filter_class.addItem("Tất cả ảnh")
        self.cmb_filter_class.addItem("Chỉ ảnh có vật thể")
        self.cmb_filter_class.addItem("Chỉ ảnh background (không nhãn)")

        colors = [
            "#eb5757", "#f2994a", "#f2c94c", "#27ae60", "#2f80ed",
            "#9b51e0", "#56ccf2", "#eb57b2", "#219653", "#f5a623",
            "#6c5ce7", "#00b894"
        ]

        row = 0
        for cid in sorted(all_counts.keys()):
            cnt = all_counts[cid]
            cname = all_names.get(cid, f"class_{cid}")
            pct = (cnt / tot_obj * 100.0) if tot_obj > 0 else 0.0
            n_imgs = all_img_counts.get(cid, 0)

            self.table_class_dist.insertRow(row)

            # Col 0: Checkbox
            item_chk = QTableWidgetItem()
            item_chk.setCheckState(Qt.CheckState.Unchecked)
            self.table_class_dist.setItem(row, 0, item_chk)

            color_hex = colors[cid % len(colors)]
            item_id = QTableWidgetItem(str(cid))
            item_id.setForeground(QColor(color_hex))
            item_id.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.table_class_dist.setItem(row, 1, item_id)

            item_name = QTableWidgetItem(cname)
            self.table_class_dist.setItem(row, 2, item_name)

            item_cnt = QTableWidgetItem(f"{cnt:,}")
            self.table_class_dist.setItem(row, 3, item_cnt)

            item_pct = QTableWidgetItem(f"{pct:.1f}%")
            item_pct.setForeground(QColor("#a6e3a1"))
            self.table_class_dist.setItem(row, 4, item_pct)

            item_imgs = QTableWidgetItem(f"{n_imgs:,} ảnh")
            self.table_class_dist.setItem(row, 5, item_imgs)

            self.cmb_filter_class.addItem(f"Class {cid}: {cname} ({cnt:,})", cid)
            row += 1

        self.cmb_filter_class.blockSignals(False)

        if hasattr(self, 'explorer_viewer'):
            self.explorer_viewer.yolo_class_names = all_names

    def filter_explorer_images(self):
        self.list_explorer_images.clear()
        self.current_filtered_images = []

        data = self.scanned_dataset_data
        if not data:
            return

        keyword = self.txt_filter_img_name.text().strip().lower()
        filter_mode = self.cmb_filter_class.currentIndex()
        selected_cid = self.cmb_filter_class.currentData()

        for ds in data.get('datasets', []):
            for rec in ds.get('images', []):
                fname = rec['filename'].lower()
                if keyword and keyword not in fname:
                    continue

                classes = rec.get('classes', [])
                if filter_mode == 1 and not classes:
                    continue
                elif filter_mode == 2 and classes:
                    continue
                elif filter_mode > 2 and selected_cid is not None:
                    if selected_cid not in classes:
                        continue

                self.current_filtered_images.append(rec)
                prefix = "🏷 " if classes else "⚪ "
                self.list_explorer_images.addItem(f"{prefix}{rec['filename']}")

        if self.current_filtered_images:
            self.list_explorer_images.setCurrentRow(0)

    def on_explorer_image_changed(self, current, previous):
        if current is not None:
            self.on_explorer_image_selected(current)

    def on_explorer_image_selected(self, item):
        row = self.list_explorer_images.currentRow()
        if row < 0 or row >= len(self.current_filtered_images):
            return

        rec = self.current_filtered_images[row]
        img_p = rec['img_path']
        lbl_p = rec.get('label_path')

        if not os.path.isfile(img_p):
            return

        img = None
        try:
            data = np.fromfile(img_p, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = cv2.imread(img_p)

        if img is None:
            return

        self.explorer_viewer.load_image_from_array(img)
        h, w = img.shape[:2]

        labels = []
        objects_table_data = []

        if lbl_p and os.path.isfile(lbl_p):
            try:
                with open(lbl_p, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        parsed = parse_yolo_line(line_str)
                        if not parsed:
                            continue
                        l_type = parsed['type']
                        data_floats = parsed['data']
                        cid = int(data_floats[0])
                        if l_type == 'bbox':
                            labels.append({'type': 'bbox', 'data': data_floats})
                            xc, yc, bw, bh = data_floats[1:5]
                            objects_table_data.append((cid, f"{xc:.3f}, {yc:.3f}, {bw:.3f}, {bh:.3f}"))
                        elif l_type == 'pose':
                            labels.append({'type': 'pose', 'data': data_floats})
                            k_count = (len(data_floats) - 5) // 3
                            objects_table_data.append((cid, f"Pose ({k_count} kpts)"))
                        elif l_type == 'poly':
                            labels.append({'type': 'poly', 'data': data_floats})
                            objects_table_data.append((cid, f"Polygon ({(len(data_floats) - 1)//2} pts)"))
            except Exception:
                pass

        all_names = self.scanned_dataset_data.get('all_class_names', {})
        self.explorer_viewer.yolo_class_names = all_names
        self.explorer_viewer.draw_yolo_labels(labels, w, h)

        classes_str = ", ".join(all_names.get(c, f"class_{c}") for c in rec.get('classes', []))
        if not classes_str:
            classes_str = "Không nhãn (Background)"
        self.lbl_explorer_img_info.setText(
            f"Kích thước: {w} × {h} px | {len(labels)} vật thể | Classes: {classes_str}"
        )

        self.table_image_objects.setRowCount(0)
        for idx, (cid, coord_str) in enumerate(objects_table_data):
            self.table_image_objects.insertRow(idx)
            self.table_image_objects.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.table_image_objects.setItem(idx, 1, QTableWidgetItem(str(cid)))
            self.table_image_objects.setItem(idx, 2, QTableWidgetItem(all_names.get(cid, f"class_{cid}")))
            self.table_image_objects.setItem(idx, 3, QTableWidgetItem(coord_str))

    # ------------------------------------------
    # SUBTAB 2: DATASET BUILDER
    # ------------------------------------------
    def setup_subtab_builder(self):
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === PANEL BÊN TRÁI: Nguồn dữ liệu & Cấu hình phân bổ ===
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(8)

        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_content = QWidget()
        lay_content = QVBoxLayout(scroll_content)
        lay_content.setSpacing(10)

        # 1. Chọn Dataset làm nguồn & Tỉ lệ lấy riêng từng nguồn
        grp_src = QGroupBox("📥 Chọn Dataset làm Nguồn & Tỉ lệ lấy riêng từng nguồn")
        lay_src = QVBoxLayout()
        self.table_builder_src = QTableWidget()
        self.table_builder_src.setColumnCount(5)
        self.table_builder_src.setHorizontalHeaderLabels([
            "Chọn", "Tên Dataset", "Tổng ảnh", "Tỉ lệ lấy (%)", "Dự kiến lấy"
        ])
        self.table_builder_src.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_builder_src.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_builder_src.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_builder_src.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_builder_src.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_builder_src.setMinimumHeight(140)
        self.table_builder_src.setMaximumHeight(220)
        lay_src.addWidget(self.table_builder_src)

        lay_src_btns = QHBoxLayout()
        btn_sel_all = QPushButton("✔ Chọn tất cả")
        btn_sel_all.clicked.connect(self.builder_select_all_sources)
        btn_desel_all = QPushButton("✖ Bỏ chọn tất cả")
        btn_desel_all.clicked.connect(self.builder_deselect_all_sources)
        lay_src_btns.addWidget(btn_sel_all)
        lay_src_btns.addWidget(btn_desel_all)
        lay_src_btns.addSpacing(15)

        lbl_quick = QLabel("⚡ Đặt % nhanh cho tất cả:")
        self.spn_builder_quick_pct = QSpinBox()
        self.spn_builder_quick_pct.setRange(0, 100)
        self.spn_builder_quick_pct.setValue(100)
        self.spn_builder_quick_pct.setSuffix(" %")
        btn_apply_quick_pct = QPushButton("Áp dụng")
        btn_apply_quick_pct.clicked.connect(self.builder_apply_quick_pct)
        lay_src_btns.addWidget(lbl_quick)
        lay_src_btns.addWidget(self.spn_builder_quick_pct)
        lay_src_btns.addWidget(btn_apply_quick_pct)
        lay_src_btns.addStretch()
        lay_src.addLayout(lay_src_btns)

        self.lbl_builder_src_summary = QLabel("Đã chọn: 0 dataset | Tổng dự kiến lấy: 0 ảnh")
        self.lbl_builder_src_summary.setStyleSheet("color: #fab387; font-weight: bold; font-size: 12px;")
        lay_src.addWidget(self.lbl_builder_src_summary)
        grp_src.setLayout(lay_src)
        lay_content.addWidget(grp_src)

        # 2. Cấu hình phân bổ / lấy mẫu Class
        grp_mode = QGroupBox("🎯 Chế độ Lấy mẫu / Phân bổ")
        lay_mode = QVBoxLayout()

        self.rb_builder_mode_pct = QRadioButton("Lấy theo tỉ lệ % riêng từng nguồn (Custom % mỗi Dataset)")
        self.rb_builder_mode_all = QRadioButton("Lấy toàn bộ ảnh từ các dataset đã chọn (100% Full Export)")
        self.rb_builder_mode_target = QRadioButton("Nhập số lượng cụ thể từng class (Target Counts)")
        self.rb_builder_mode_pct.setChecked(True)

        self.builder_mode_group = QButtonGroup(self)
        self.builder_mode_group.addButton(self.rb_builder_mode_pct, 0)
        self.builder_mode_group.addButton(self.rb_builder_mode_all, 1)
        self.builder_mode_group.addButton(self.rb_builder_mode_target, 2)
        self.builder_mode_group.idToggled.connect(lambda _id, _chk: self.on_builder_mode_changed() if _chk else None)

        lay_mode.addWidget(self.rb_builder_mode_pct)

        self.widget_pct_hint = QLabel("💡 Bạn có thể thiết lập % riêng cho từng dataset tại cột 'Tỉ lệ lấy (%)' trong bảng nguồn ở trên, hoặc dùng công cụ đặt nhanh %.")
        self.widget_pct_hint.setStyleSheet("color: #a6e3a1; font-style: italic; margin-left: 20px;")
        self.widget_pct_hint.setWordWrap(True)
        lay_mode.addWidget(self.widget_pct_hint)

        lay_mode.addWidget(self.rb_builder_mode_all)
        lay_mode.addWidget(self.rb_builder_mode_target)

        # Bảng target counts
        self.widget_target_table = QWidget()
        lay_tt = QVBoxLayout(self.widget_target_table)
        lay_tt.setContentsMargins(0, 0, 0, 0)
        lbl_hint = QLabel("Nhập số lượng vật thể cần lấy cho mỗi class:")
        lbl_hint.setStyleSheet("color: #a6adc8; font-size: 11px;")
        lay_tt.addWidget(lbl_hint)

        self.table_builder_targets = QTableWidget()
        self.table_builder_targets.setColumnCount(4)
        self.table_builder_targets.setHorizontalHeaderLabels(["ID", "Tên Class", "Có sẵn", "Cần lấy (Target)"])
        self.table_builder_targets.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_builder_targets.horizontalHeader().setStretchLastSection(True)
        self.table_builder_targets.setMaximumHeight(160)
        lay_tt.addWidget(self.table_builder_targets)
        self.widget_target_table.setVisible(False)
        lay_mode.addWidget(self.widget_target_table)

        # Background images
        lay_bg = QHBoxLayout()
        self.chk_builder_include_bg = QCheckBox("Lấy thêm ảnh Background (không nhãn):")
        self.chk_builder_include_bg.setChecked(False)
        self.spn_builder_bg_count = QSpinBox()
        self.spn_builder_bg_count.setRange(0, 100000)
        self.spn_builder_bg_count.setValue(50)
        self.spn_builder_bg_count.setSuffix(" ảnh")
        self.spn_builder_bg_count.setEnabled(False)
        self.chk_builder_include_bg.toggled.connect(self.spn_builder_bg_count.setEnabled)
        lay_bg.addWidget(self.chk_builder_include_bg)
        lay_bg.addWidget(self.spn_builder_bg_count)
        lay_mode.addLayout(lay_bg)

        grp_mode.setLayout(lay_mode)
        lay_content.addWidget(grp_mode)

        # 3. Thư mục Dataset Mới
        grp_dest = QGroupBox("📤 Thư mục Dataset Mới & data.yaml")
        lay_dest = QGridLayout()

        lay_dest.addWidget(QLabel("📂 Thư mục gốc:"), 0, 0)
        self.txt_builder_output_dir = QLineEdit()
        self.txt_builder_output_dir.setPlaceholderText("Chọn thư mục lưu dataset mới...")
        lay_dest.addWidget(self.txt_builder_output_dir, 0, 1)
        btn_out_b = QPushButton("Browse...")
        btn_out_b.clicked.connect(self.browse_builder_output)
        lay_dest.addWidget(btn_out_b, 0, 2)

        lay_dest.addWidget(QLabel("📁 Tên Dataset:"), 1, 0)
        self.txt_builder_dataset_name = QLineEdit()
        self.txt_builder_dataset_name.setText("custom_yolo_dataset")
        lay_dest.addWidget(self.txt_builder_dataset_name, 1, 1, 1, 2)

        self.chk_builder_create_yaml = QCheckBox("Tự động tạo file data.yaml chuẩn YOLO (Ultralytics compatible)")
        self.chk_builder_create_yaml.setChecked(True)
        self.chk_builder_create_yaml.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        lay_dest.addWidget(self.chk_builder_create_yaml, 2, 0, 1, 3)

        grp_dest.setLayout(lay_dest)
        lay_content.addWidget(grp_dest)

        scroll_left.setWidget(scroll_content)
        left_lay.addWidget(scroll_left)
        splitter.addWidget(left_widget)

        # === PANEL BÊN PHẢI: Split Train/Val/Test & Tiến trình thực thi ===
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(8)

        # 4. Phân chia Train / Val / Test
        grp_split = QGroupBox("🍰 Phân chia tỉ lệ Train / Val / Test")
        lay_split = QVBoxLayout()

        lay_sp_row = QHBoxLayout()
        lay_sp_row.addWidget(QLabel("Train:"))
        self.spn_split_train = QSpinBox()
        self.spn_split_train.setRange(0, 100)
        self.spn_split_train.setValue(80)
        self.spn_split_train.setSuffix(" %")
        self.spn_split_train.valueChanged.connect(self.update_split_sum_label)
        lay_sp_row.addWidget(self.spn_split_train)

        lay_sp_row.addWidget(QLabel("Val:"))
        self.spn_split_val = QSpinBox()
        self.spn_split_val.setRange(0, 100)
        self.spn_split_val.setValue(10)
        self.spn_split_val.setSuffix(" %")
        self.spn_split_val.valueChanged.connect(self.update_split_sum_label)
        lay_sp_row.addWidget(self.spn_split_val)

        lay_sp_row.addWidget(QLabel("Test:"))
        self.spn_split_test = QSpinBox()
        self.spn_split_test.setRange(0, 100)
        self.spn_split_test.setValue(10)
        self.spn_split_test.setSuffix(" %")
        self.spn_split_test.valueChanged.connect(self.update_split_sum_label)
        lay_sp_row.addWidget(self.spn_split_test)

        self.lbl_split_sum = QLabel("Tổng: 100% ✅")
        self.lbl_split_sum.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        lay_sp_row.addWidget(self.lbl_split_sum)
        lay_split.addLayout(lay_sp_row)

        lay_presets = QHBoxLayout()
        lay_presets.addWidget(QLabel("Presets nhanh:"))
        btn_pre1 = QPushButton("80 / 10 / 10")
        btn_pre1.clicked.connect(lambda: self.set_split_preset(80, 10, 10))
        btn_pre2 = QPushButton("70 / 20 / 10")
        btn_pre2.clicked.connect(lambda: self.set_split_preset(70, 20, 10))
        btn_pre3 = QPushButton("90 / 10 / 0")
        btn_pre3.clicked.connect(lambda: self.set_split_preset(90, 10, 0))
        btn_pre4 = QPushButton("100 / 0 / 0")
        btn_pre4.clicked.connect(lambda: self.set_split_preset(100, 0, 0))
        lay_presets.addWidget(btn_pre1)
        lay_presets.addWidget(btn_pre2)
        lay_presets.addWidget(btn_pre3)
        lay_presets.addWidget(btn_pre4)
        lay_presets.addStretch()
        lay_split.addLayout(lay_presets)

        grp_split.setLayout(lay_split)
        right_lay.addWidget(grp_split)

        # 5. Thực thi & Tiến trình
        grp_exec = QGroupBox("⚡ Thực thi & Tiến trình")
        lay_exec = QVBoxLayout()

        lay_exec_btns = QHBoxLayout()
        self.btn_run_build = QPushButton("🚀 BẮT ĐẦU TẠO DATASET MỚI")
        self.btn_run_build.setMinimumHeight(40)
        self.btn_run_build.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #11998e, stop:1 #38ef7d); "
            "color: #1e1e2e; font-weight: bold; font-size: 13px; border-radius: 6px; }"
            "QPushButton:hover { background: #38ef7d; }"
        )
        self.btn_run_build.clicked.connect(self.toggle_run_build_dataset)
        lay_exec_btns.addWidget(self.btn_run_build)

        self.btn_stop_build = QPushButton("⏹ Dừng")
        self.btn_stop_build.setMinimumHeight(40)
        self.btn_stop_build.setEnabled(False)
        self.btn_stop_build.setStyleSheet(
            "QPushButton { background: #e74c3c; color: white; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: #c0392b; }"
        )
        self.btn_stop_build.clicked.connect(self.stop_run_build_dataset)
        lay_exec_btns.addWidget(self.btn_stop_build)
        lay_exec.addLayout(lay_exec_btns)

        self.builder_progress_bar = QProgressBar()
        self.builder_progress_bar.setValue(0)
        self.builder_progress_bar.setMinimumHeight(20)
        lay_exec.addWidget(self.builder_progress_bar)

        lay_exec.addWidget(QLabel("📝 Log tiến trình:"))
        self.txt_builder_log = QTextEdit()
        self.txt_builder_log.setReadOnly(True)
        self.txt_builder_log.setMinimumHeight(240)
        lay_exec.addWidget(self.txt_builder_log)

        grp_exec.setLayout(lay_exec)
        right_lay.addWidget(grp_exec)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)

        layout.addWidget(splitter)
        self.subtab_builder.setLayout(layout)

    def refresh_builder_sources_ui(self):
        """Cập nhật bảng danh sách các dataset nguồn và bảng target counts"""
        data = self.scanned_dataset_data
        datasets = data.get('datasets', [])

        # Ghi nhớ trạng thái checkbox và % đã chọn trước đó nếu có
        prev_states = {}
        for r_data in self.builder_source_rows:
            p = r_data.get('path')
            chk_widget = r_data.get('checkbox')
            spn_widget = r_data.get('spinbox')
            if p and chk_widget and spn_widget:
                prev_states[p] = (chk_widget.isChecked(), spn_widget.value())

        self.table_builder_src.blockSignals(True)
        self.table_builder_src.setRowCount(0)
        self.builder_source_rows.clear()

        for r_idx, ds in enumerate(datasets):
            self.table_builder_src.insertRow(r_idx)
            ds_path = ds.get('path', '')
            saved_chk, saved_pct = prev_states.get(ds_path, (True, 100))

            # Col 0: Checkbox
            chk = QCheckBox()
            chk.setChecked(saved_chk)
            w_chk = QWidget()
            l_chk = QHBoxLayout(w_chk)
            l_chk.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l_chk.setContentsMargins(0, 0, 0, 0)
            l_chk.addWidget(chk)
            self.table_builder_src.setCellWidget(r_idx, 0, w_chk)

            # Col 1: Tên Dataset
            item_name = QTableWidgetItem(ds.get('name', ''))
            item_name.setToolTip(f"Đường dẫn: {ds.get('path', '')}")
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_builder_src.setItem(r_idx, 1, item_name)

            # Col 2: Tổng ảnh
            img_cnt = ds.get('img_count', 0)
            item_cnt = QTableWidgetItem(f"{img_cnt:,}")
            item_cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_cnt.setFlags(item_cnt.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_builder_src.setItem(r_idx, 2, item_cnt)

            # Col 3: Tỉ lệ lấy (%)
            spn_pct = QSpinBox()
            spn_pct.setRange(0, 100)
            spn_pct.setValue(saved_pct)
            spn_pct.setSuffix(" %")
            spn_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spn_pct.setStyleSheet("font-weight: bold; color: #a6e3a1;")
            self.table_builder_src.setCellWidget(r_idx, 3, spn_pct)

            # Col 4: Dự kiến lấy
            item_exp = QTableWidgetItem(f"{img_cnt:,} ảnh")
            item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_exp.setFlags(item_exp.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item_exp.setForeground(QColor("#fab387"))
            self.table_builder_src.setItem(r_idx, 4, item_exp)

            # Kết nối sự kiện
            chk.toggled.connect(self.on_builder_source_toggled)
            spn_pct.valueChanged.connect(self.on_builder_source_toggled)

            self.builder_source_rows.append({
                'ds': ds,
                'path': ds.get('path', ''),
                'name': ds.get('name', ''),
                'total': img_cnt,
                'checkbox': chk,
                'spinbox': spn_pct,
                'item_exp': item_exp
            })

        self.table_builder_src.blockSignals(False)
        self.on_builder_source_toggled()

    def on_builder_source_toggled(self):
        """Khi checkbox hoặc spinbox % của dataset nguồn thay đổi: tính toán lại số ảnh và bảng targets"""
        m_id = self.builder_mode_group.checkedId()
        # 0: pct, 1: all, 2: target_counts

        checked_paths = set()
        total_selected_datasets = 0
        total_selected_images = 0
        total_source_images = 0
        avail_classes = {}

        for row_data in self.builder_source_rows:
            chk = row_data['checkbox']
            spn = row_data['spinbox']
            item_exp = row_data['item_exp']
            total = row_data['total']
            ds = row_data['ds']

            if chk.isChecked():
                checked_paths.add(row_data['path'])
                total_selected_datasets += 1
                total_source_images += total

                if m_id == 1: # 100% All
                    spn.setEnabled(False)
                    expected = total
                    item_exp.setText(f"{total:,} ảnh (100%)")
                elif m_id == 2: # Target counts
                    spn.setEnabled(False)
                    expected = total
                    item_exp.setText("Theo targets")
                else: # 0: Custom % mỗi dataset
                    spn.setEnabled(True)
                    pct = spn.value()
                    expected = int(round(total * pct / 100.0))
                    if pct > 0 and expected == 0 and total > 0:
                        expected = 1
                    item_exp.setText(f"{expected:,} / {total:,} ảnh ({pct}%)")

                total_selected_images += expected

                for cid, cnt in ds.get('class_counts', {}).items():
                    avail_classes[cid] = avail_classes.get(cid, 0) + cnt
            else:
                spn.setEnabled(False)
                item_exp.setText("0 ảnh (bỏ qua)")

        if m_id == 2:
            self.lbl_builder_src_summary.setText(
                f"Đã chọn: {total_selected_datasets}/{len(self.builder_source_rows)} dataset "
                f"(Tổng nguồn: {total_source_images:,} ảnh | Lấy mẫu theo Target Counts bên dưới)"
            )
        else:
            self.lbl_builder_src_summary.setText(
                f"Đã chọn: {total_selected_datasets}/{len(self.builder_source_rows)} dataset | "
                f"Tổng dự kiến lấy: {total_selected_images:,} ảnh (tổng nguồn: {total_source_images:,} ảnh)"
            )

        # Lưu lại các target counts người dùng đã nhập trước đó
        prev_targets = {}
        for r in range(self.table_builder_targets.rowCount()):
            c_item = self.table_builder_targets.item(r, 0)
            w_spn = self.table_builder_targets.cellWidget(r, 3)
            if c_item and isinstance(w_spn, QSpinBox):
                try:
                    prev_targets[int(c_item.text())] = w_spn.value()
                except ValueError:
                    pass

        all_names = self.scanned_dataset_data.get('all_class_names', {})
        self.table_builder_targets.setRowCount(0)
        row = 0
        for cid in sorted(avail_classes.keys()):
            cnt = avail_classes[cid]
            cname = all_names.get(cid, f"class_{cid}")

            self.table_builder_targets.insertRow(row)
            self.table_builder_targets.setItem(row, 0, QTableWidgetItem(str(cid)))
            self.table_builder_targets.setItem(row, 1, QTableWidgetItem(cname))
            self.table_builder_targets.setItem(row, 2, QTableWidgetItem(f"{cnt:,}"))

            spn_t = QSpinBox()
            spn_t.setRange(0, cnt)
            if cid in prev_targets:
                spn_t.setValue(min(cnt, prev_targets[cid]))
            else:
                spn_t.setValue(cnt)
            self.table_builder_targets.setCellWidget(row, 3, spn_t)
            row += 1

    def builder_select_all_sources(self):
        for r in self.builder_source_rows:
            r['checkbox'].setChecked(True)

    def builder_deselect_all_sources(self):
        for r in self.builder_source_rows:
            r['checkbox'].setChecked(False)

    def builder_apply_quick_pct(self):
        val = self.spn_builder_quick_pct.value()
        for r in self.builder_source_rows:
            if r['checkbox'].isChecked():
                r['spinbox'].setValue(val)

    def on_builder_mode_changed(self):
        m_id = self.builder_mode_group.checkedId()
        self.widget_target_table.setVisible(m_id == 2)
        self.widget_pct_hint.setVisible(m_id == 0)
        self.on_builder_source_toggled()

    def update_split_sum_label(self):
        tr = self.spn_split_train.value()
        val = self.spn_split_val.value()
        te = self.spn_split_test.value()
        s = tr + val + te
        if s == 100:
            self.lbl_split_sum.setText("Tổng: 100% ✅")
            self.lbl_split_sum.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        else:
            self.lbl_split_sum.setText(f"Tổng: {s}% ⚠ Cần 100%")
            self.lbl_split_sum.setStyleSheet("color: #f38ba8; font-weight: bold;")

    def set_split_preset(self, tr, val, te):
        self.spn_split_train.setValue(tr)
        self.spn_split_val.setValue(val)
        self.spn_split_test.setValue(te)

    def browse_builder_output(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu Dataset Mới")
        if path:
            self.txt_builder_output_dir.setText(path)

    def toggle_run_build_dataset(self):
        out_base = self.txt_builder_output_dir.text().strip()
        if not out_base:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn Thư mục lưu Dataset Mới!")
            return

        ds_name = self.txt_builder_dataset_name.text().strip()
        if not ds_name:
            ds_name = "custom_yolo_dataset"
        final_out_dir = os.path.join(out_base, ds_name)

        tr = self.spn_split_train.value()
        val = self.spn_split_val.value()
        te = self.spn_split_test.value()
        if tr + val + te != 100:
            QMessageBox.warning(self, "Lỗi", "Tổng tỉ lệ Train + Val + Test phải bằng 100%!")
            return

        checked_paths = set()
        source_percentages = {}
        for r in self.builder_source_rows:
            if r['checkbox'].isChecked():
                p = r['path']
                checked_paths.add(p)
                source_percentages[p] = r['spinbox'].value()

        if not checked_paths:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ít nhất 1 Dataset làm nguồn!")
            return

        candidate_images = []
        for ds in self.scanned_dataset_data.get('datasets', []):
            if ds['path'] in checked_paths:
                candidate_images.extend(ds.get('images', []))

        if not candidate_images:
            QMessageBox.warning(self, "Lỗi", "Không có file ảnh nào trong các dataset nguồn đã chọn!")
            return

        m_id = self.builder_mode_group.checkedId()
        mode_str = 'percentage'
        target_counts = {}
        if m_id == 1:
            mode_str = 'all'
        elif m_id == 2:
            mode_str = 'target_counts'
            for r in range(self.table_builder_targets.rowCount()):
                cid_item = self.table_builder_targets.item(r, 0)
                spn = self.table_builder_targets.cellWidget(r, 3)
                if cid_item and spn:
                    cid = int(cid_item.text())
                    target_counts[cid] = spn.value()
        else:
            mode_str = 'percentage'

        cfg = {
            'available_images': candidate_images,
            'source_percentages': source_percentages,
            'mode': mode_str,
            'target_counts': target_counts,
            'split_ratios': {'train': tr, 'val': val, 'test': te},
            'output_dir': final_out_dir,
            'class_names': self.scanned_dataset_data.get('all_class_names', {}),
            'include_bg': self.chk_builder_include_bg.isChecked(),
            'bg_count': self.spn_builder_bg_count.value(),
            'create_yaml': self.chk_builder_create_yaml.isChecked()
        }

        self.btn_run_build.setEnabled(False)
        self.btn_stop_build.setEnabled(True)
        self.builder_progress_bar.setValue(0)
        self.txt_builder_log.clear()

        self.dataset_builder_worker = DatasetBuilderWorkerThread(cfg)
        self.dataset_builder_worker.log_signal.connect(self.on_builder_log)
        self.dataset_builder_worker.progress_signal.connect(self.on_builder_progress)
        self.dataset_builder_worker.finished_signal.connect(self.on_builder_finished)
        self.dataset_builder_worker.start()

    def stop_run_build_dataset(self):
        if self.dataset_builder_worker and self.dataset_builder_worker.isRunning():
            self.dataset_builder_worker.stop()

    def on_builder_log(self, msg):
        self.txt_builder_log.append(msg)
        self.txt_builder_log.moveCursor(QTextCursor.MoveOperation.End)

    def on_builder_progress(self, current, total):
        pct = int((current / total) * 100) if total > 0 else 0
        self.builder_progress_bar.setValue(pct)

    def on_builder_finished(self, success, msg, summary):
        self.btn_run_build.setEnabled(True)
        self.btn_stop_build.setEnabled(False)

        if success:
            QMessageBox.information(
                self, "Thành công",
                f"🎉 {msg}\n\n"
                f"• Tổng số ảnh: {summary.get('total_images', 0)}\n"
                f"• Train: {summary.get('train_count', 0)} ảnh\n"
                f"• Val: {summary.get('val_count', 0)} ảnh\n"
                f"• Test: {summary.get('test_count', 0)} ảnh\n"
                f"• File data.yaml đã được tạo sẵn sàng huấn luyện!\n\n"
                f"Thư mục lưu:\n{summary.get('output_dir', '')}"
            )
        else:
            QMessageBox.warning(self, "Thông báo", f"Quá trình tạo dataset: {msg}")

    def load_settings(self):
        """Load cài đặt đã lưu từ lần chạy trước"""
        qs = QSettings("VisionAI", "AutoDefect")
        in_folder = qs.value("input_folder", "")
        self.txt_input_folder.setText(in_folder)
        if in_folder and os.path.isdir(in_folder):
            files = self._get_image_files(in_folder)
            self.lbl_file_count.setText(f"{len(files)} ảnh")
        self.txt_output_folder.setText(qs.value("output_folder", ""))
        self.txt_labels_folder.setText(qs.value("labels_folder", ""))
        classes = qs.value("yolo_classes", None)
        self.cmb_yolo_class.clear()
        if classes and isinstance(classes, list):
            self.cmb_yolo_class.addItems(classes)
        self._update_yolo_class_names()
        if hasattr(self, 'txt_prs_input_folder'):
            self.txt_prs_input_folder.setText(qs.value("prs_input_folder", ""))
            if self.txt_prs_input_folder.text():
                self.refresh_prs_images_list()
        if hasattr(self, 'txt_prs_output_folder'):
            self.txt_prs_output_folder.setText(qs.value("prs_output_folder", ""))
        if hasattr(self, 'chk_prs_zone_as_seg'):
            val = qs.value("prs_zone_as_seg", False, type=bool)
            self.chk_prs_zone_as_seg.setChecked(val)
        if hasattr(self, 'chk_dzone_fallback'):
            val_fb = qs.value("prs_dzone_fallback", True, type=bool)
            self.chk_dzone_fallback.setChecked(val_fb)
        if hasattr(self, 'cmb_search_dir'):
            idx_dir = qs.value("prs_search_dir", 0, type=int)
            self.cmb_search_dir.setCurrentIndex(idx_dir)
        if hasattr(self, 'spn_search_dist'):
            dist_val = qs.value("prs_search_dist", 100, type=int)
            self.spn_search_dist.setValue(dist_val)
        # Tab 5 Crop settings
        if hasattr(self, 'txt_crop_input_folder'):
            self.txt_crop_input_folder.setText(qs.value("crop_input_folder", ""))
            if self.txt_crop_input_folder.text():
                self.refresh_crop_images_list()
        if hasattr(self, 'txt_crop_output_folder'):
            self.txt_crop_output_folder.setText(qs.value("crop_output_folder", ""))
        if hasattr(self, 'spn_crop_margin'):
            self.spn_crop_margin.setValue(qs.value("crop_margin", 0, type=int))
        if hasattr(self, 'cmb_crop_format'):
            self.cmb_crop_format.setCurrentIndex(qs.value("crop_format", 0, type=int))
        if hasattr(self, 'cmb_crop_naming'):
            self.cmb_crop_naming.setCurrentIndex(qs.value("crop_naming", 0, type=int))
        if hasattr(self, 'chk_crop_use_prs'):
            self.chk_crop_use_prs.setChecked(qs.value("crop_use_prs", True, type=bool))
        # Tab 6 Dataset Manager settings
        saved_ds_paths = qs.value("dataset_source_paths", None)
        if saved_ds_paths and isinstance(saved_ds_paths, list):
            self.dataset_source_paths = saved_ds_paths
            if hasattr(self, 'list_dataset_sources'):
                self.list_dataset_sources.clear()
                for p in self.dataset_source_paths:
                    self.list_dataset_sources.addItem(p)
                if self.dataset_source_paths:
                    self.scan_all_datasets()
        if hasattr(self, 'txt_builder_output_dir'):
            self.txt_builder_output_dir.setText(qs.value("builder_output_dir", ""))
        if hasattr(self, 'txt_builder_dataset_name'):
            self.txt_builder_dataset_name.setText(qs.value("builder_dataset_name", "custom_yolo_dataset"))

    def save_settings(self):
        """Lưu cài đặt hiện tại"""
        qs = QSettings("VisionAI", "AutoDefect")
        qs.setValue("input_folder", self.txt_input_folder.text())
        qs.setValue("output_folder", self.txt_output_folder.text())
        qs.setValue("labels_folder", self.txt_labels_folder.text())
        classes = [self.cmb_yolo_class.itemText(i) for i in range(self.cmb_yolo_class.count())]
        qs.setValue("yolo_classes", classes)
        if hasattr(self, 'txt_prs_input_folder'):
            qs.setValue("prs_input_folder", self.txt_prs_input_folder.text())
        if hasattr(self, 'txt_prs_output_folder'):
            qs.setValue("prs_output_folder", self.txt_prs_output_folder.text())
        if hasattr(self, 'chk_prs_zone_as_seg'):
            qs.setValue("prs_zone_as_seg", self.chk_prs_zone_as_seg.isChecked())
        if hasattr(self, 'chk_dzone_fallback'):
            qs.setValue("prs_dzone_fallback", self.chk_dzone_fallback.isChecked())
        if hasattr(self, 'cmb_search_dir'):
            qs.setValue("prs_search_dir", self.cmb_search_dir.currentIndex())
        if hasattr(self, 'spn_search_dist'):
            qs.setValue("prs_search_dist", self.spn_search_dist.value())
        # Tab 5 Crop settings
        if hasattr(self, 'txt_crop_input_folder'):
            qs.setValue("crop_input_folder", self.txt_crop_input_folder.text())
        if hasattr(self, 'txt_crop_output_folder'):
            qs.setValue("crop_output_folder", self.txt_crop_output_folder.text())
        if hasattr(self, 'spn_crop_margin'):
            qs.setValue("crop_margin", self.spn_crop_margin.value())
        if hasattr(self, 'cmb_crop_format'):
            qs.setValue("crop_format", self.cmb_crop_format.currentIndex())
        if hasattr(self, 'cmb_crop_naming'):
            qs.setValue("crop_naming", self.cmb_crop_naming.currentIndex())
        if hasattr(self, 'chk_crop_use_prs'):
            qs.setValue("crop_use_prs", self.chk_crop_use_prs.isChecked())
        # Tab 6 Dataset Manager settings
        if hasattr(self, 'dataset_source_paths'):
            qs.setValue("dataset_source_paths", self.dataset_source_paths)
        if hasattr(self, 'txt_builder_output_dir'):
            qs.setValue("builder_output_dir", self.txt_builder_output_dir.text())
        if hasattr(self, 'txt_builder_dataset_name'):
            qs.setValue("builder_dataset_name", self.txt_builder_dataset_name.text())

    def closeEvent(self, event):
        """Đảm bảo tắt worker khi đóng app"""
        self.save_settings()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        if hasattr(self, 'prs_worker') and self.prs_worker and self.prs_worker.isRunning():
            self.prs_worker.stop()
            self.prs_worker.wait()
        if hasattr(self, 'crop_worker') and self.crop_worker and self.crop_worker.isRunning():
            self.crop_worker.stop()
            self.crop_worker.wait()
        if hasattr(self, 'dataset_scan_worker') and self.dataset_scan_worker and self.dataset_scan_worker.isRunning():
            self.dataset_scan_worker.stop()
            self.dataset_scan_worker.wait()
        if hasattr(self, 'dataset_builder_worker') and self.dataset_builder_worker and self.dataset_builder_worker.isRunning():
            self.dataset_builder_worker.stop()
            self.dataset_builder_worker.wait()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            if hasattr(self, 'tabs') and self.tabs.currentIndex() == 2:  # Tab 3: Manual YOLO Labeling
                selected_yolo = [
                    item for item in self.yolo_viewer.scene.selectedItems()
                    if hasattr(self.yolo_viewer, 'yolo_items') and item in self.yolo_viewer.yolo_items
                ]
                if selected_yolo:
                    for item in selected_yolo:
                        self.yolo_viewer.scene.removeItem(item)
                        self.yolo_viewer.yolo_items.remove(item)
                    event.accept()
                    return
                elif getattr(self, 'current_yolo_img_path', '') and os.path.exists(self.current_yolo_img_path):
                    self.delete_current_yolo_file()
                    event.accept()
                    return
        super().keyPressEvent(event)
