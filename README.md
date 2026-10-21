# Defect & Dataset Studio AI 🎨🤖
**DBC AI Studio & Synthetic Defect Generator for Industrial Computer Vision**

A comprehensive, production-grade desktop application for synthetic defect generation, automated PRS alignment, manual/auto YOLO annotation (BBox, Segmentation, Pose), and dataset lifecycle management (YOLOv8, YOLOv11).

---

## 🌟 Key Features

### 📋 1. Recipe & Zone Setup
- **Master Image & Position Reference System (PRS)**: Dual-point PRS (PRS 1 & PRS 2) with Search ROI and Pattern ROI for high-precision alignment.
- **Defect Zones**: Interactive rectangular and polygon zones defining where defects can appear.
- **Ignore Zones**: Mask out protected areas where no defect should be drawn.
- **Keypoint Setup**: Define keypoint templates for YOLO-Pose datasets.
- **Recipe Persistence**: Full save/load support for JSON recipe files.

### 🚀 2. Batch Generate Defects (Synthetic Generation)
- **10 Defect Brush Models**: Crack, Scratch, Contamination, Dust, Fiber, Spot, Bubble, Burn, Dent, and Real Stamp Bank.
- **Auto Surface Detection**: Detect dark/bright regions using Otsu or thresholding with safety margins.
- **Procedural Fiber & Stamp Bank**: Procedural multi-segment branching curves and real-world stamp blending.
- **YOLO Export**: Automated bounding box, polygon segmentation, and pose label generation.

### ✍️ 3. Manual YOLO Labeling
- Full manual annotation editor for Bounding Box, Segmentation Polygon, and Pose Keypoints.
- Fast review workflow: Press `Del` to delete selected annotations or permanently delete bad images + corresponding `.txt` labels from disk with automatic navigation to the next image.

### 🎯 4. Auto PRS & Labeling (Seg / Pose)
- Automated contour extraction and PRS compensation on batch images.
- Automatic generation of segmentation masks and pose keypoints based on master recipes.

### ✂️ 5. Batch Crop Zones
- Automatically align target images via PRS and batch-crop component regions/defect zones into dedicated folders.

### 📊 6. Dataset Manager
- **Dataset Explorer**: Scan and analyze single or multi-source datasets with class distributions, object counts, and background detection.
- **Bulk Class Deletion**: Safely delete unwanted classes across selected dataset sources with automatic index realignment (0..N-1), `data.yaml` synchronization, and backup.
- **Dataset Builder**: Sample from multiple source datasets by percentage or target counts, split into Train/Val/Test, and export standard `data.yaml`.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows 10/11

### Dependencies
```bash
pip install PyQt6 opencv-python numpy
```

### Running the Application
```bash
python main.py
```

---

## 📁 Project Structure

```
├── main.py                     # Application entry point
├── main_window.py              # Main GUI controller (6 Tabs)
├── custom_widgets.py           # Custom PyQt6 widgets (ImageViewer, ResizableRect/Polygon)
├── brush_engine.py             # Procedural defect simulation engine
├── worker.py                   # Multi-threaded background workers
├── utils.py                    # Utility functions (Unicode cv2 I/O, YOLO parsing, etc.)
├── assets/                     # Stamp textures and asset samples
├── recipes/                    # Sample recipe JSON files
└── HANDOVER_CONTEXT.md         # Comprehensive project handover & technical specification
```

---

## 📄 License
Internal industrial computer vision & AI research tool. All rights reserved.
