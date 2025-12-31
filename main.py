import sys
from collections import deque
import time

import cv2
import numpy as np
import pyvirtualcam
import colorsys
import math
import json
import os
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from PyQt5.QtCore import (Qt, QTimer, QPoint, QRect, QSize, 
                             QPropertyAnimation, QEasingCurve, QPointF)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
                             QPushButton, QCheckBox, QLineEdit, QProgressBar,
                              QComboBox, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QFormLayout, QRadioButton, QGridLayout,
                             QFileDialog, QStatusBar, QMessageBox,
                             QDesktopWidget, QScrollArea)
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush


class PreviewLabel(QLabel):
    """An interactive label for dragging calibration pins on the camera feed."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = [[100, 100], [540, 100], [540, 380], [100, 380]]
        self.dragging_idx = -1
        self.manual_mode = False
        self.parent_win = None

    def mousePressEvent(self, event):
        if not self.manual_mode: return
        pos = event.pos()
        for i, (px, py) in enumerate(self.points):
            # Simple distance check for handles (20px radius)
            if math.sqrt((pos.x() - px)**2 + (pos.y() - py)**2) < 20:
                self.dragging_idx = i
                break

    def mouseMoveEvent(self, event):
        if self.dragging_idx != -1:
            px, py = event.pos().x(), event.pos().y()
            # Constrain to label bounds
            px = max(0, min(px, self.width()))
            py = max(0, min(py, self.height()))
            self.points[self.dragging_idx] = [px, py]
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_idx = -1
        if self.parent_win:
            self.parent_win.save_preview_mapping()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.manual_mode: return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw lines
        painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
        pts = [QPoint(int(p[0]), int(p[1])) for p in self.points]
        for i in range(4):
            painter.drawLine(pts[i], pts[(i+1)%4])
            
        # Draw handles
        for i, (px, py) in enumerate(self.points):
            color = QColor(255, 0, 255) if i == self.dragging_idx else QColor(0, 255, 255)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.white, 1))
            painter.drawEllipse(QPoint(int(px), int(py)), 10, 10)
            painter.drawText(int(px+12), int(py+12), f"P{i+1}")
        painter.end()


def calculate_iou(boxA, boxB):
    # box = (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


class FaceTrack:
    """Helper class to maintain state for a single tracked face."""
    def __init__(self, box, frame):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, box)
        self.history = deque([box], maxlen=5)
        self.last_box = box
        self.last_overlay_bbox = None  # (x1, y1, x2, y2)
        self.failed_frames = 0

    def update(self, frame):
        success, box = self.tracker.update(frame)
        if success:
            self.last_box = tuple(map(int, box))
            self.history.append(self.last_box)
            self.failed_frames = 0
            return True, self.last_box
        else:
            self.failed_frames += 1
            return False, None

    def get_avg_box(self):
        if not self.history:
            return self.last_box
        return tuple(np.mean(self.history, axis=0).astype(int))


class ProjectorWindow(QMainWindow):
    """A secondary window for professional projection mapping."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Projector Output")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black;")
        
        self.points = self.load_points()
        self.dragging_idx = -1
        self.win_dragging = False
        self.drag_pos = None
        self.calibration_mode = False
        self.show_pattern = False
        
        # Load stored homography if it exists
        config = MainWindow.get_projector_config_static()
        h_mat = config.get("homography")
        self.homography_matrix = np.array(h_mat, dtype=np.float32) if h_mat else None
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.last_frame = None

    def load_points(self):
        config = MainWindow.get_projector_config_static()
        return config.get("points", [[50, 50], [400, 50], [400, 300], [50, 300]])

    def save_config(self, geometry=None):
        config = MainWindow.get_projector_config_static()
        config["points"] = self.points
        if geometry:
            config["geometry"] = geometry
        MainWindow.write_projector_config_static(config)

    def set_calibration_mode(self, enabled):
        self.calibration_mode = enabled
        if not enabled:
            self.save_config()
        self.update_display()

    def set_show_pattern(self, enabled):
        self.show_pattern = enabled
        self.update_display()

    def draw_chessboard(self, w, h):
        """Generates a high-contrast calibration pattern."""
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        rows, cols = 6, 9 # Internal corners are (cols-1)x(rows-1)
        sq_w, sq_h = w // cols, h // rows
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    cv2.rectangle(canvas, (j*sq_w, i*sq_h), ((j+1)*sq_w, (i+1)*sq_h), (255, 255, 255), -1)
        return canvas

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Handle points calibration first
            if self.calibration_mode:
                pos = event.pos()
                for i, (px, py) in enumerate(self.points):
                    if math.hypot(pos.x() - px, pos.y() - py) < 20:
                        self.dragging_idx = i
                        return
            
            # If frameless, allow dragging window by background
            if self.windowFlags() & Qt.FramelessWindowHint:
                self.win_dragging = True
                self.drag_pos = event.globalPos() - self.pos()

    def mouseMoveEvent(self, event):
        if self.dragging_idx != -1:
            self.points[self.dragging_idx] = [event.pos().x(), event.pos().y()]
            self.update_display()
        elif self.win_dragging:
            self.move(event.globalPos() - self.drag_pos)

    def mouseReleaseEvent(self, event):
        if self.win_dragging:
            self.win_dragging = False
            # Save new position
            geom = [self.x(), self.y(), self.width(), self.height()]
            self.save_config(geometry=geom)
        self.dragging_idx = -1

    def mouseDoubleClickEvent(self, event):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def update_frame(self, frame):
        self.last_frame = frame
        self.update_display()

    def update_display(self):
        wh, ww = self.height(), self.width()
        
        if self.show_pattern:
            # Show calibration chessboard
            pattern = self.draw_chessboard(ww, wh)
            rgb = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, ww, wh, ww * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))
            return

        if self.last_frame is None: 
            self.label.setPixmap(QPixmap()) # Set empty pixmap if no frame
            return
        
        # Note: macOS/OpenCV may emit a deprecation warning about 'AVCaptureDeviceTypeExternal'.
        # This is a harmless internal notification from Apple/OpenCV and does not affect performance.
        
        h, w = self.last_frame.shape[:2]
        
        # --- NEW: Direct Mode ---
        # If the frame already matches our dimensions, assume it's pre-warped/native
        if (w == ww and h == wh) and not self.calibration_mode:
            rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, ww, wh, ww * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))
            return

        # Ensure we have precisely 4 points for the manual warp fallback
        if not hasattr(self, 'points') or not isinstance(self.points, list) or len(self.points) != 4:
            self.points = [[50, 50], [ww-50, 50], [ww-50, wh-50], [50, wh-50]]

        # Source points (the full frame)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        # Destination points (warped corners)
        dst = np.array(self.points, dtype=np.float32)
        
        try:
            if self.homography_matrix is not None and not self.calibration_mode:
                # Use pre-calculated homography if available and not in calibration mode
                M = self.homography_matrix
            else:
                M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(self.last_frame, M, (ww, wh))
        except Exception as e:
            # Fallback for degenerate points
            warped = cv2.resize(self.last_frame, (ww, wh))
            print(f"Warp Error: {e}")
        
        # Convert to QImage
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, ww, wh, ww * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        # Overlay calibration handles if in manual calibration mode
        if self.calibration_mode:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(0, 255, 0), 3)) # Green pen for lines and circles
            
            # Draw connecting lines
            # Ensure p[0], p[1] are converted to int for QPoint
            pts = [QPoint(int(p[0]), int(p[1])) for p in self.points]
            painter.drawLine(pts[0], pts[1])
            painter.drawLine(pts[1], pts[2])
            painter.drawLine(pts[2], pts[3])
            painter.drawLine(pts[3], pts[0])

            # Draw handles
            for i, (px, py) in enumerate(self.points):
                color = QColor(255, 0, 255) if i == self.dragging_idx else QColor(0, 255, 0)
                painter.setBrush(color)
                painter.drawEllipse(int(px-10), int(py-10), 20, 20) # Larger handles
                painter.setPen(QPen(QColor(255, 255, 255), 1)) # White pen for text
                painter.drawText(int(px+15), int(py+15), f"C{i+1}") # Label corners
            painter.end()

        self.label.setPixmap(pix)


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Tracking Overlay Control")
        self.setGeometry(100, 100, 1200, 700)

        # --- State and Processing Setup ---
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.tracking_active = True
        self.mirror_mode = True
        self.projection_mode = False
        self.live_mode = False  # Master Switch
        self.last_face_width = 0
        self.raw_frame_size = (640, 480) # Default, will be updated
        self.offset_x, self.offset_y = 0, 0
        self.trackers = []  # List of FaceTrack objects
        self.frame_count = 0 
        
        # --- Virtual Camera ---
        self.virtual_cam = None
        self.virtual_cam_active = False

        # --- Overlay Properties ---
        self.overlay_mode = 'Text'
        self.overlay_text = "Tracking..."
        self.font_size = 40
        self.font_color = (255, 255, 255)
        self.overlay_image = None
        self.image_scale = 1.0
        self.scale_with_face = False
        
        # GIF properties
        self.is_gif = False
        self.gif_frames = []
        self.gif_idx = 0
        
        # Effects State
        self.effect_type = "None" # None, Rainbow, Pulse, Neon
        self.effect_speed = 1.0
        self.effect_phase = 0.0
        
        # Font State
        self.pil_font = None 
        self.font_path = None # Default system font if None
        self.font_name_display = "Default"
        
        # Load a default font if possible, else standard
        try:
             # Try a common mac font initially
             self.pil_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.font_size)
             self.font_name_display = "Helvetica"
        except IOError:
             self.pil_font = ImageFont.load_default()
             self.font_name_display = "Default"

        self.setup_ui()
        
        # --- Projector Window ---
        self.projector_window = None

        # --- Timer Setup ---
        # Initialize camera on main thread
        self.cap = cv2.VideoCapture(0)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # ~33 FPS

    def setup_ui(self):
        """Initializes the user interface."""
        main_layout = QHBoxLayout()
        self.video_label = PreviewLabel("Connecting to camera...")
        self.video_label.parent_win = self
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        main_layout.addWidget(self.video_label, 3)

        control_layout = QVBoxLayout()
        self.setup_core_controls(control_layout)
        self.setup_position_controls(control_layout)
        self.setup_overlay_type_controls(control_layout)
        self.setup_text_controls(control_layout)
        self.setup_image_controls(control_layout)
        control_layout.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(control_widget)
        scroll_area.setFixedWidth(350)
        main_layout.addWidget(scroll_area, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setStatusBar(QStatusBar(self))

    def setup_core_controls(self, layout):
        core_group = QGroupBox("Core Controls")
        core_layout = QVBoxLayout()
        
        # --- NEW: PERFORMANCE SECTION (TOP) ---
        perf_group = QGroupBox("Performance Mode")
        perf_layout = QVBoxLayout()
        
        self.mode_label = QLabel("MODE: SETUP (Muted)")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #fbc02d; margin-bottom: 5px;")
        perf_layout.addWidget(self.mode_label)

        self.go_live_btn = QPushButton("GO LIVE")
        self.go_live_btn.setCheckable(True)
        self.go_live_btn.setMinimumHeight(70)
        self.go_live_btn.setStyleSheet("""
            QPushButton { 
                background-color: #388e3c; color: white; font-size: 26px; font-weight: bold; border-radius: 12px;
            }
            QPushButton:checked {
                background-color: #d32f2f;
                border: 3px solid #ffcdd2;
            }
        """)
        self.go_live_btn.toggled.connect(self.toggle_live_mode)
        perf_layout.addWidget(self.go_live_btn)

        prox_layout = QHBoxLayout()
        prox_layout.addWidget(QLabel("Depth Focus:"))
        self.proximity_bar = QProgressBar()
        self.proximity_bar.setRange(0, 100)
        self.proximity_bar.setTextVisible(True)
        prox_layout.addWidget(self.proximity_bar)
        perf_layout.addLayout(prox_layout)
        
        perf_group.setLayout(perf_layout)
        core_layout.addWidget(perf_group)

        # --- EXISTING: SETUP & PROJECTOR (MIDDLE) ---
        proj_panel = QGroupBox("Calibration & Output")
        proj_grid = QGridLayout()
        
        self.open_proj_btn = QPushButton("Open Window")
        self.open_proj_btn.setToolTip("Open the secondary projector window.")
        self.open_proj_btn.clicked.connect(self.toggle_projector_window)
        proj_grid.addWidget(self.open_proj_btn, 0, 0)

        self.jump_proj_btn = QPushButton("Project NOW!")
        self.jump_proj_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.jump_proj_btn.clicked.connect(self.project_now)
        proj_grid.addWidget(self.jump_proj_btn, 0, 1)

        self.pattern_checkbox = QCheckBox("Show Grid Pattern")
        self.pattern_checkbox.setEnabled(False)
        self.pattern_checkbox.toggled.connect(self.toggle_pattern_mode)
        proj_grid.addWidget(self.pattern_checkbox, 1, 0)

        self.auto_align_btn = QPushButton("Auto-Align Grid")
        self.auto_align_btn.setEnabled(False)
        self.auto_align_btn.clicked.connect(self.run_auto_calibration)
        proj_grid.addWidget(self.auto_align_btn, 1, 1)

        self.calib_checkbox = QCheckBox("Manual Mapping")
        self.calib_checkbox.setEnabled(False)
        self.calib_checkbox.toggled.connect(self.toggle_calibration_mode)
        proj_grid.addWidget(self.calib_checkbox, 2, 0)
        
        self.reset_calib_btn = QPushButton("Reset Mapping")
        self.reset_calib_btn.clicked.connect(self.reset_calibration)
        proj_grid.addWidget(self.reset_calib_btn, 2, 1)

        proj_panel.setLayout(proj_grid)
        core_layout.addWidget(proj_panel)

        # Screen Selection
        screen_layout = QHBoxLayout()
        screen_layout.addWidget(QLabel("Target Screen:"))
        self.screen_combo = QComboBox()
        self.refresh_screens()
        self.screen_combo.currentIndexChanged.connect(self.move_projector_to_screen)
        screen_layout.addWidget(self.screen_combo)
        core_layout.addLayout(screen_layout)

        mirror_checkbox = QCheckBox("Mirror Camera Feed")
        mirror_checkbox.setToolTip("Flip the camera image horizontally.")
        mirror_checkbox.toggled.connect(self.set_mirror_mode)
        core_layout.addWidget(mirror_checkbox)

        projection_checkbox = QCheckBox("Projection Mode")
        tooltip = "Show overlay on a black background for projection."
        projection_checkbox.setToolTip(tooltip)
        projection_checkbox.toggled.connect(self.set_projection_mode)
        core_layout.addWidget(projection_checkbox)

        self.vcam_checkbox = QCheckBox("Virtual Camera Output")
        self.vcam_checkbox.setToolTip("Stream output to a virtual camera device (e.g. OBS Camera).")
        self.vcam_checkbox.toggled.connect(self.toggle_virtual_camera)
        core_layout.addWidget(self.vcam_checkbox)

        reset_pos_button = QPushButton("Reset Tracker & Position")
        tooltip = "Reset overlay position and re-initialize tracker."
        reset_pos_button.setToolTip(tooltip)
        reset_pos_button.clicked.connect(self.reset_all)
        core_layout.addWidget(reset_pos_button)

        core_group.setLayout(core_layout)
        layout.addWidget(core_group)
        
        # Load preview points from config if they exist
        config = self.get_projector_config()
        preview_pts = config.get("preview_points")
        if preview_pts:
            self.video_label.points = preview_pts

    def setup_position_controls(self, layout):
        pos_group = QGroupBox("Position Offsets")
        pos_layout = QFormLayout()

        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-1000, 1000)
        self.offset_x_spin.setSingleStep(10)
        self.offset_x_spin.setValue(0)
        self.offset_x_spin.setToolTip("Horizontal offset (pixels). Negative=Left, Positive=Right.")
        self.offset_x_spin.valueChanged.connect(self.set_offset_x)
        pos_layout.addRow("Offset X:", self.offset_x_spin)

        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-1000, 1000)
        self.offset_y_spin.setSingleStep(10)
        self.offset_y_spin.setValue(0)
        self.offset_y_spin.setToolTip("Vertical offset (pixels). Negative=Up, Positive=Down.")
        self.offset_y_spin.valueChanged.connect(self.set_offset_y)
        pos_layout.addRow("Offset Y:", self.offset_y_spin)

        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

    def setup_overlay_type_controls(self, layout):
        type_group = QGroupBox("Overlay Type")
        type_layout = QHBoxLayout()
        self.text_radio = QRadioButton("Text")
        self.text_radio.setChecked(True)
        self.text_radio.toggled.connect(self.set_overlay_type)
        self.image_radio = QRadioButton("Image")
        type_layout.addWidget(self.text_radio)
        type_layout.addWidget(self.image_radio)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

    def setup_text_controls(self, layout):
        self.text_group = QGroupBox("Text Overlay Settings")
        form_layout = QFormLayout()

        self.text_input = QLineEdit(self.overlay_text)
        self.text_input.setToolTip("The text to display on the overlay.")
        self.text_input.textChanged.connect(self.set_overlay_text)
        form_layout.addRow("Text:", self.text_input)

        # Font Selection
        font_layout = QHBoxLayout()
        self.load_font_btn = QPushButton("Load Font...")
        self.load_font_btn.clicked.connect(self.load_font)
        self.font_label = QLabel(self.font_name_display)
        font_layout.addWidget(self.load_font_btn)
        font_layout.addWidget(self.font_label)
        form_layout.addRow("Font:", font_layout)

        self.size_spinbox = QSpinBox()
        self.size_spinbox.setToolTip("The size of the overlay text.")
        self.size_spinbox.setRange(10, 500)
        self.size_spinbox.setValue(self.font_size)
        self.size_spinbox.valueChanged.connect(self.set_font_size)
        form_layout.addRow("Size:", self.size_spinbox)
        
        # Effects
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["None", "Rainbow", "Pulse", "Neon"])
        self.effect_combo.currentTextChanged.connect(self.set_effect_type)
        form_layout.addRow("Effect:", self.effect_combo)
        
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(self.effect_speed)
        self.speed_spin.valueChanged.connect(self.set_effect_speed)
        form_layout.addRow("Eff. Speed:", self.speed_spin)

        self.text_group.setLayout(form_layout)
        layout.addWidget(self.text_group)

    def setup_image_controls(self, layout):
        self.image_group = QGroupBox("Image Overlay Settings")
        image_form_layout = QFormLayout()

        load_image_button = QPushButton("Load Image...")
        load_image_button.setToolTip("Open a file dialog to select an image.")
        load_image_button.clicked.connect(self.load_image)
        self.image_path_label = QLabel("No image loaded.")
        self.image_path_label.setWordWrap(True)
        image_form_layout.addRow(load_image_button, self.image_path_label)

        self.image_scale_spinbox = QDoubleSpinBox()
        tooltip = "The size of the image, relative to face width."
        self.image_scale_spinbox.setToolTip(tooltip)
        self.image_scale_spinbox.setRange(0.1, 10.0)
        self.image_scale_spinbox.setSingleStep(0.1)
        self.image_scale_spinbox.setValue(self.image_scale)
        self.image_scale_spinbox.valueChanged.connect(self.set_image_scale)
        image_form_layout.addRow("Scale:", self.image_scale_spinbox)

        self.scale_toggle = QCheckBox("Scale with Face Size")
        self.scale_toggle.setChecked(self.scale_with_face)
        self.scale_toggle.toggled.connect(self.set_scale_with_face)
        image_form_layout.addRow(self.scale_toggle)

        self.image_group.setLayout(image_form_layout)
        self.image_group.setEnabled(False)
        layout.addWidget(self.image_group)

    # --- Slot methods for controls ---
    def set_mirror_mode(self, checked): self.mirror_mode = checked
    def set_projection_mode(self, checked): self.projection_mode = checked
    def set_offset_x(self, val): self.offset_x = val
    def set_offset_y(self, val): self.offset_y = val
    def set_overlay_text(self, text): self.overlay_text = text
    def set_font_size(self, val): 
        self.font_size = val
        self.update_pil_font()
    def set_effect_type(self, val): self.effect_type = val
    def set_effect_speed(self, val): self.effect_speed = val
    def set_image_scale(self, val): self.image_scale = val
    def set_scale_with_face(self, checked): self.scale_with_face = checked

    def update_pil_font(self):
        try:
            path = self.font_path if self.font_path else "/System/Library/Fonts/Helvetica.ttc" #"Arial.ttf" #fallback
            self.pil_font = ImageFont.truetype(path, self.font_size)
        except:
             try:
                 self.pil_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.font_size)
             except:
                 self.pil_font = ImageFont.load_default()

    def load_image(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "All Images (*.png *.jpg *.jpeg *.gif);;PNG (*.png);;JPG (*.jpg *.jpeg);;GIF (*.gif)")
        if fpath:
            try:
                if fpath.lower().endswith(".gif"):
                    pil_img = Image.open(fpath)
                    frames = []
                    for frame in ImageSequence.Iterator(pil_img):
                        # Convert to RGBA then to Numpy BGR(A)
                        frame = frame.convert("RGBA")
                        cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
                        frames.append(cv_frame)
                    
                    if frames:
                        self.gif_frames = frames
                        self.is_gif = True
                        self.gif_idx = 0
                        self.overlay_image = frames[0] # Set first frame as preview or fallback
                        
                        fname = fpath.split('/')[-1]
                        self.image_path_label.setText(f"{fname} (GIF)")
                        self.statusBar().showMessage(f"Loaded GIF {fname}", 3000)
                    else:
                        raise ValueError("No frames found in GIF")
                else:
                    self.overlay_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    self.is_gif = False
                    if self.overlay_image is not None:
                        fname = fpath.split('/')[-1]
                        self.image_path_label.setText(fname)
                        self.statusBar().showMessage(f"Loaded {fname}", 3000)
                    else:
                        self.image_path_label.setText("Error loading image.")
                        self.statusBar().showMessage("Failed to load image.", 3000)
            except Exception as e:
                self.image_path_label.setText("Error loading image.")
                self.statusBar().showMessage(f"Failed to load image: {e}", 3000)
    def load_font(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Font", "", "Font Files (*.ttf *.otf *.ttc)")
        if fpath:
            self.font_path = fpath
            self.font_name_display = fpath.split('/')[-1]
            self.font_label.setText(self.font_name_display)
            self.update_pil_font()

    def toggle_virtual_camera(self, checked):
        if checked:
            try:
                # Attempt to create a virtual camera instance. 
                # Note: We don't know the exact resolution yet, but we'll adapt or re-init if needed,
                # or typically init with the first frame. For now, we'll lazy-load in update_image
                # or init with a standard size if we can get it from the cap.
                # Simpler approach: Just set the flag and init in update_image if None.
                self.virtual_cam_active = True
            except Exception as e:
                self.statusBar().showMessage(f"Error starting Virtual Cam: {e}", 4000)
                self.vcam_checkbox.setChecked(False)
        else:
            self.virtual_cam_active = False
            if self.virtual_cam:
                self.virtual_cam.close()
                self.virtual_cam = None
                self.statusBar().showMessage("Virtual Camera stopped.", 2000)

    def toggle_projector_window(self):
        if self.projector_window is None:
            self.projector_window = ProjectorWindow()
            
            # Load stored geometry if exists
            config = MainWindow.get_projector_config_static()
            geom = config.get("geometry")
            if geom:
                self.projector_window.setGeometry(geom[0], geom[1], geom[2], geom[3])
            
            # If no stored geom or on a screen that doesn't exist, use default
            current_screen_idx = self.screen_combo.currentIndex()
            if not geom and current_screen_idx != -1:
                self.move_projector_to_screen(current_screen_idx)
            else:
                self.projector_window.showFullScreen()

            self.open_proj_btn.setText("Close Window")
            self.open_proj_btn.setStyleSheet("background-color: #d32f2f; color: white;")
            self.calib_checkbox.setEnabled(True)
            self.pattern_checkbox.setEnabled(True)
            self.auto_align_btn.setEnabled(True)
            self.statusBar().showMessage("Projector window opened in Kiosk Mode.", 2000)
        else:
            self.projector_window.close()
            self.projector_window = None
            self.open_proj_btn.setText("Open Window")
            self.open_proj_btn.setStyleSheet("")
            self.calib_checkbox.setChecked(False)
            self.calib_checkbox.setEnabled(False)
            self.pattern_checkbox.setChecked(False)
            self.pattern_checkbox.setEnabled(False)
            self.auto_align_btn.setEnabled(False)
            self.statusBar().showMessage("Projector window closed.", 2000)

    def toggle_live_mode(self, checked):
        self.live_mode = checked
        if checked:
            self.go_live_btn.setText("STOP PERFORMANCE")
            self.mode_label.setText("MODE: LIVE PERFORMANCE")
            self.mode_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #d32f2f;")
            self.statusBar().showMessage("LIVE MODE ACTIVE - Tracking enabled.", 4000)
            # Hide setup tools if visible
            self.pattern_checkbox.setChecked(False)
        else:
            self.go_live_btn.setText("GO LIVE")
            self.mode_label.setText("MODE: SETUP (Muted)")
            self.mode_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #fbc02d;")
            self.statusBar().showMessage("Setup Mode - Tracking muted.", 4000)

    def toggle_pattern_mode(self, checked):
        if self.projector_window:
            self.projector_window.set_show_pattern(checked)

    def run_auto_calibration(self):
        """Performs automated camera-projector alignment using chessboard pattern."""
        if not self.projector_window: return
        
        # 1. Show the pattern
        self.pattern_checkbox.setChecked(True)
        QApplication.processEvents() # Ensure window updates
        time.sleep(0.5) # Wait for display to refresh
        
        # 2. Capture a few frames to be sure we see the projected pattern
        for _ in range(5):
            ret, cv_img = self.cap.read()
            if self.mirror_mode:
                cv_img = cv2.flip(cv_img, 1)

        if not ret:
            self.statusBar().showMessage("Auto-Calibration Failed: Could not capture frame.", 3000)
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Find corners in the camera image
        # Projector uses 6 rows, 9 columns -> Internal corners (8x5)
        pattern_size = (8, 5) 
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if found:
            # Sub-pixel refinement for higher accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 4. Define ground-truth corners in projector-space (Full projector window)
            pw, ph = self.projector_window.width(), self.projector_window.height()
            rows, cols = 6, 9
            # Internal intersections calculation (8x5 grid)
            # We want to map intersections 1..8 and rows 1..5
            sq_w = pw / cols
            sq_h = ph / rows
            proj_corners = []
            for i in range(1, 6): # Internal row intersections
                for j in range(1, 9): # Internal column intersections
                    proj_corners.append([j * sq_w, i * sq_h])
            
            proj_pts = np.array(proj_corners, dtype=np.float32)
            cam_pts = corners.reshape(-1, 2)
            
            # 5. Calculate Homography (Mapping Camera Pixels -> Projector Pixels)
            H, _ = cv2.findHomography(cam_pts, proj_pts, cv2.RANSAC, 5.0)
            
            # 6. Save and apply
            config = self.get_projector_config()
            config["homography"] = H.tolist()
            self.write_projector_config_static(config)
            
            self.projector_window.homography_matrix = H
            
            # --- NEW: Update Preview Handles ---
            # Map projector corners (0,0 -> pw,ph) back to UI preview space
            H_inv = np.linalg.inv(H)
            proj_corners_full = np.array([[0,0], [pw,0], [pw,ph], [0,ph]], dtype=np.float32).reshape(-1, 1, 2)
            cam_corners_full = cv2.perspectiveTransform(proj_corners_full, H_inv).reshape(-1, 2)
            
            # Convert Cam-Space to UI-Space
            lw, lh = self.video_label.width(), self.video_label.height()
            fw, fh = self.raw_frame_size
            ui_pts = []
            for cx, cy in cam_corners_full:
                ux = cx * (lw / fw) if fw > 0 else cx
                uy = cy * (lh / fh) if fh > 0 else cy
                ui_pts.append([int(ux), int(uy)])
            
            self.video_label.points = ui_pts
            self.video_label.update()
            
            self.pattern_checkbox.setChecked(False)
            self.statusBar().showMessage("Auto-Alignment Success!", 5000)
            QMessageBox.information(self, "Success", 
                "Projector spatially aligned!\n\nI found all 40 reference points and updated your handles. You can now fine-tune them if needed.")
        else:
            self.statusBar().showMessage("Auto-Calibration Failed: Chessboard not found.", 5000)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Calibration Help")
            msg.setText("I couldn't find the calibration pattern.")
            msg.setInformativeText("Tips:\n1. Ensure the camera sees the WHOLE grid.\n2. Avoid glare or very bright reflections.\n3. Make sure the grid is reasonably flat.")
            msg.exec_()

    def project_now(self):
        """One-click to find the best projector screen and go fullscreen."""
        screens = QApplication.screens()
        if len(screens) < 2:
            QMessageBox.information(self, "No Projector", "Only one screen detected. I'll open it here anyway!")
        
        # Open window
        if not self.projector_window:
            self.toggle_projector_window()
        
        # Pick the second screen if it exists (usually the projector)
        target_idx = 1 if len(screens) > 1 else 0
        self.screen_combo.setCurrentIndex(target_idx)
        self.move_projector_to_screen(target_idx)
        self.statusBar().showMessage(f"Projecting to {screens[target_idx].name()}", 3000)

    def refresh_screens(self):
        self.screen_combo.clear()
        screens = QApplication.screens()
        for i, s in enumerate(screens):
            self.screen_combo.addItem(f"Screen {i}: {s.name()} ({s.size().width()}x{s.size().height()})")
        
        # Try to restore last screen index from config
        config = self.get_projector_config()
        idx = config.get("monitor_index", 0)
        if idx < self.screen_combo.count():
            self.screen_combo.blockSignals(True)
            self.screen_combo.setCurrentIndex(idx)
            self.screen_combo.blockSignals(False)

    @staticmethod
    def get_projector_config_static():
        """Globally safe config loader."""
        if os.path.exists("mapping_config.json"):
            try:
                with open("mapping_config.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return {"points": data}
                    if isinstance(data, dict):
                        return data
            except: pass
        return {"points": [[50, 50], [400, 50], [400, 300], [50, 300]]}

    @staticmethod
    def write_projector_config_static(config):
        """Globally safe config writer."""
        try:
            with open("mapping_config.json", "w") as f:
                json.dump(config, f)
        except: pass

    def get_projector_config(self):
        return self.get_projector_config_static()

    def move_projector_to_screen(self, idx):
        if idx < 0: return
        screens = QApplication.screens()
        if idx < len(screens):
            screen_geo = screens[idx].geometry()
            if self.projector_window:
                # We move and resize to fill that screen but keep it "windowed" so it has decorations
                self.projector_window.showNormal()
                self.projector_window.setGeometry(screen_geo)
                self.projector_window.show()
            
            # Save monitor index
            config = self.get_projector_config()
            if not isinstance(config, dict): 
                config = {"points": [[50, 50], [400, 50], [400, 300], [50, 300]]}
            
            config["monitor_index"] = idx
            self.write_projector_config_static(config)

    def reset_calibration(self):
        """Clears homography and resets manual mapping points to defaults."""
        if not self.projector_window: return
        
        reply = QMessageBox.question(self, 'Reset Calibration', 
                                    "Are you sure you want to clear all calibration data?", 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Clear homography
            self.projector_window.homography_matrix = None
            
            # Reset points to generic rectangle
            ww, wh = self.projector_window.width(), self.projector_window.height()
            self.projector_window.points = [[50, 50], [ww-50, 50], [ww-50, wh-50], [50, wh-50]]
            
            # Save to config
            config = self.get_projector_config()
            if "homography" in config: del config["homography"]
            config["points"] = self.projector_window.points
            self.write_projector_config_static(config)
            
            self.projector_window.update_display()
            self.statusBar().showMessage("Calibration reset to defaults.", 3000)

    def toggle_calibration_mode(self, checked):
        if self.projector_window:
            self.projector_window.set_calibration_mode(checked)
        self.video_label.manual_mode = checked
        self.video_label.update()
        msg = "Mapping Mode Active - Drag Green Corners" if checked else "Mapping Mode Disabled"
        self.statusBar().showMessage(msg, 3000)

    def save_preview_mapping(self):
        """Converts UI preview handles to actual camera-space and calculates Homography."""
        if not self.projector_window: return
        
        # 1. Get scaling factors
        lw, lh = self.video_label.width(), self.video_label.height()
        fw, fh = self.raw_frame_size
        
        scale_x = fw / lw if lw > 0 else 1.0
        scale_y = fh / lh if lh > 0 else 1.0
        
        # 2. Map UI points to Camera-Space
        cam_pts = []
        for px, py in self.video_label.points:
            cam_pts.append([px * scale_x, py * scale_y])
        cam_pts = np.array(cam_pts, dtype=np.float32)

        # 3. Define Projector Corners (Full Screen)
        pw, ph = self.projector_window.width(), self.projector_window.height()
        proj_pts = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)

        # 4. Calculate Homography
        H, _ = cv2.findHomography(cam_pts, proj_pts)
        
        # 5. Save and apply
        self.projector_window.homography_matrix = H
        config = self.get_projector_config()
        config["homography"] = H.tolist()
        config["preview_points"] = self.video_label.points
        self.write_projector_config_static(config)
        self.statusBar().showMessage("Spatial mapping updated from Preview.", 2000)

    def reset_all(self):
        self.offset_x_spin.setValue(0)
        self.offset_y_spin.setValue(0)
        self.trackers = []
        self.statusBar().showMessage("All trackers and position reset.", 3000)

    def set_overlay_type(self):
        self.overlay_mode = "Image" if self.image_radio.isChecked() else "Text"
        self.text_group.setEnabled(self.overlay_mode == "Text")
        self.image_group.setEnabled(self.overlay_mode == "Image")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        if self.virtual_cam:
            self.virtual_cam.close()
        if self.projector_window:
            self.projector_window.close()
        event.accept()

    # --- Drawing and Processing Methods ---
    def draw_text_overlay_pil(self, img_pil, draw, face_box, scale_factor=1.0):
        """Draws text on a PIL image object."""
        x, y, w, h = face_box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Determine Color
        draw_color = self.font_color
        if self.effect_type == "Rainbow":
            hue = (self.effect_phase * 0.1) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
            draw_color = (int(r*255), int(g*255), int(b*255))
            
        # Determine Size/Font (Pulse effect)
        current_font = self.pil_font
        if self.effect_type == "Pulse":
            scale = 1.0 + 0.2 * math.sin(self.effect_phase)
            try:
                new_size = int(self.font_size * scale_factor * scale)
                current_font = ImageFont.truetype(self.font_path or "/System/Library/Fonts/Helvetica.ttc", max(10, new_size))
            except:
                pass
                
        # Get Size
        try:
             bbox = draw.textbbox((0, 0), self.overlay_text, font=current_font)
             text_w = bbox[2] - bbox[0]
             text_h = bbox[3] - bbox[1]
        except AttributeError:
             text_w, text_h = draw.textsize(self.overlay_text, font=current_font)

        # Anchor
        base_pos_x = center_x - text_w // 2
        base_pos_y = center_y - text_h // 2
        final_pos = (base_pos_x + int(self.offset_x * scale_factor), 
                     base_pos_y + int(self.offset_y * scale_factor))
        
        # Store overlay bbox for masking (approximate text extent)
        self.last_overlay_bbox = (final_pos[0], final_pos[1], final_pos[0] + text_w, final_pos[1] + text_h)
        
        # Draw
        if self.effect_type == "Neon":
             outline_color = (0, 255, 255)
             if(self.effect_phase % 20 > 10): outline_color = (255, 0, 255)
             thickness = 4
             for ox in range(-thickness, thickness+1):
                 for oy in range(-thickness, thickness+1):
                      draw.text((final_pos[0]+ox, final_pos[1]+oy), self.overlay_text, font=current_font, fill=outline_color)
             draw.text(final_pos, self.overlay_text, font=current_font, fill=(255, 255, 255))
        else:
             draw.text(final_pos, self.overlay_text, font=current_font, fill=draw_color)

    def draw_image_overlay(self, canvas, face_box, scale_factor=1.0):
        """Draws the selected image/gif on the canvas."""
        x, y, w, h = face_box
        
        # Determine center and offset
        center_x, center_y = x + w // 2, y + h // 2
        offset_x_scaled, offset_y_scaled = int(self.offset_x * scale_factor), int(self.offset_y * scale_factor)
        
        # Determine image to draw
        current_image = self.overlay_image
        if self.is_gif and self.gif_frames:
            self.gif_idx = (self.gif_idx + 1) % len(self.gif_frames)
            current_image = self.gif_frames[self.gif_idx]

        if current_image is None: return

        overlay_h, overlay_w = current_image.shape[:2]
        
        if self.scale_with_face:
            scaled_w = int(w * self.image_scale)
        else:
            # Fixed scale: using a reference size (e.g., 200px) multiplied by scale and scale_factor
            scaled_w = int(200 * self.image_scale * scale_factor)
            
        scaled_h = int(overlay_h * (scaled_w / overlay_w))
        resized = cv2.resize(current_image, (scaled_w, scaled_h))

        pos_x = center_x - scaled_w // 2 + offset_x_scaled
        pos_y = center_y - scaled_h // 2 + offset_y_scaled

        # Get the region of interest and handle boundary conditions
        x1, y1 = max(pos_x, 0), max(pos_y, 0)
        x2 = min(pos_x + scaled_w, canvas.shape[1])
        y2 = min(pos_y + scaled_h, canvas.shape[0])
        overlay_x1 = max(0, -pos_x)
        overlay_y1 = max(0, -pos_y)
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        if (x2 > x1) and (y2 > y1):
            # Store actual drawn bbox for masking
            for track in self.trackers:
                if track.get_avg_box() == face_box:
                    track.last_overlay_bbox = (x1, y1, x2, y2)
            
            roi = canvas[y1:y2, x1:x2]
            part = resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
            if part.shape[2] == 4:  # Handle alpha channel
                alpha = part[:, :, 3] / 255.0
                alpha_inv = 1.0 - alpha
                for c in range(3):
                    roi[:, :, c] = (alpha * part[:, :, c] +
                                    alpha_inv * roi[:, :, c])
            else:  # No alpha channel
                roi[:] = part

    def update_frame(self):
        ret, cv_img = self.cap.read()
        if not ret:
            return
        self.process_image(cv_img)

    def process_image(self, cv_img):
        """The main processing loop for each frame."""
        h, w = cv_img.shape[:2]
        self.raw_frame_size = (w, h)
        
        if self.mirror_mode:
            cv_img = cv2.flip(cv_img, 1)

        # 1. Capture/Setup Analysis (Always runs for Proximity/Alignment Feed)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Calibration Debug View (Green dots when Show Pattern is on)
        if self.pattern_checkbox.isChecked():
            found, corners = cv2.findChessboardCorners(gray, (8, 5), 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
            if found:
                cv2.drawChessboardCorners(cv_img, (8, 5), corners, found)
                self.statusBar().showMessage("Pattern DETECTED - Ready for Auto-Align", 1000)
            else:
                self.statusBar().showMessage("Pattern NOT found - Adjust camera/projector", 1000)

        # 2. Tracking & Distance Estimation
        tracked_boxes = self.track_or_detect(cv_img)
        
        # Update Proximity Meter
        if tracked_boxes:
            # We use the first face for distance estimation
            _, _, fw, _ = tracked_boxes[0]
            self.last_face_width = fw
            
            # Simple heuristic: 120-180px width is usually "Optimal" for desk/stage
            # We scale to 0-100 where 50 is center (perfect)
            prox_val = int(np.clip((fw / 300.0) * 100, 0, 100))
            self.proximity_bar.setValue(prox_val)
            if fw < 100:
                self.proximity_bar.setFormat("Too Far")
                self.proximity_bar.setStyleSheet("QProgressBar::chunk { background-color: #fbc02d; }")
            elif fw > 220:
                self.proximity_bar.setFormat("Too Close")
                self.proximity_bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
            else:
                self.proximity_bar.setFormat("OPTIMAL DISTANCE")
                self.proximity_bar.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
        else:
            self.proximity_bar.setValue(0)
            self.proximity_bar.setFormat("No Face Detected")

        # 3. Layer Composition
        # Create a "Clean" overlay buffer (black background) for Main UI (Alpha-blending)
        # And a "Projector" native buffer (matching projector resolution)
        overlay_canvas = np.zeros_like(cv_img)
        
        pw, ph = (640, 480) # Defaults
        if self.projector_window:
            pw, ph = self.projector_window.width(), self.projector_window.height()
        proj_overlay = np.zeros((ph, pw, 3), dtype=np.uint8)

        # --- PERFORMANCE LOGIC ---
        # Only process overlays if GO LIVE is active
        if self.live_mode and tracked_boxes:
            # Get mapping points for clipping
            quad_pts = np.array(self.video_label.points, dtype=np.float32)
            
            # Prep points for transformation
            # We scale UI points to camera-space for clipping/transform
            lw, lh = self.video_label.width(), self.video_label.height()
            fw, fh = self.raw_frame_size
            sx, sy = fw/lw, fh/lh
            cam_quad = (quad_pts * [sx, sy]).astype(np.float32)

            for track_box in tracked_boxes:
                tx, ty, tw, th = track_box
                cx, cy = tx + tw/2, ty + th/2 # Face Center
                
                # A. Clipping Check: Is face center inside the projection zone?
                # pointPolygonTest expects points in float32
                dist = cv2.pointPolygonTest(cam_quad, (float(cx), float(cy)), False)
                if dist < 0: continue # Outside
                
                # B. Transform Center to Projector Space
                if self.projector_window and self.projector_window.homography_matrix is not None:
                    # perspectiveTransform expects a list of points (shape 1, N, 2)
                    pts = np.array([[[cx, cy]]], dtype=np.float32)
                    trans_pts = cv2.perspectiveTransform(pts, self.projector_window.homography_matrix)
                    ux, uy = trans_pts[0,0]
                else:
                    # Fallback to simple scaling if no homography
                    ux, uy = cx * (pw/fw), cy * (ph/fh)

                # C. Draw Content
                # For Main UI (camera-space preview)
                if self.overlay_mode == 'Text':
                    self.effect_phase += 0.1 * self.effect_speed
                    # We draw on PIL for high-quality text
                    img_ui = Image.fromarray(cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2RGB))
                    img_proj = Image.fromarray(cv2.cvtColor(proj_overlay, cv2.COLOR_BGR2RGB))
                    
                    draw_ui = ImageDraw.Draw(img_ui)
                    draw_proj = ImageDraw.Draw(img_proj)
                    
                    # Compute scale factor for projector vs camera (based on width)
                    p_scale = pw / fw if fw > 0 else 1.0

                    # Draw on UI (camera space, scale=1.0)
                    self.draw_text_overlay_pil(img_ui, draw_ui, track_box, scale_factor=1.0)
                    # Draw on Projector (projector space, scale=p_scale)
                    proj_box = (int(ux - tw/2 * p_scale), int(uy - th/2 * p_scale), 
                                int(tw * p_scale), int(th * p_scale))
                    self.draw_text_overlay_pil(img_proj, draw_proj, proj_box, scale_factor=p_scale)
                    
                    overlay_canvas[:] = cv2.cvtColor(np.array(img_ui), cv2.COLOR_RGB2BGR)
                    proj_overlay[:] = cv2.cvtColor(np.array(img_proj), cv2.COLOR_RGB2BGR)
                    
                elif self.overlay_mode == 'Image' and self.overlay_image is not None:
                    p_scale = pw / fw if fw > 0 else 1.0
                    # Draw on UI
                    self.draw_image_overlay(overlay_canvas, track_box, scale_factor=1.0)
                    # Draw on Projector
                    proj_box = (int(ux - tw/2 * p_scale), int(uy - th/2 * p_scale), 
                                int(tw * p_scale), int(th * p_scale))
                    self.draw_image_overlay(proj_overlay, proj_box, scale_factor=p_scale)

        # 4. Composition for Main UI / Virtual Cam
        if self.projection_mode:
            output_canvas = overlay_canvas.copy()
        else:
            output_canvas = cv_img.copy()
            mask = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY) > 0
            output_canvas[mask] = overlay_canvas[mask]
            
            # Draw projection quad border for feedback (Setup Mode Only)
            if not self.live_mode:
                lw, lh = self.video_label.width(), self.video_label.height()
                fw, fh = self.raw_frame_size
                sx, sy = fw/lw, fh/lh
                pts_vis = (np.array(self.video_label.points) * [sx, sy]).astype(np.int32)
                cv2.polylines(output_canvas, [pts_vis], True, (255, 255, 0), 2)

            # Draw debug boxes (only in Setup Mode)
            if not self.live_mode:
                for track_box in tracked_boxes:
                    x, y, w, h = track_box
                    cv2.rectangle(output_canvas, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # 5. Output to Virtual Camera
        if self.virtual_cam_active:
            try:
                ov_h, ov_w = output_canvas.shape[:2]
                if self.virtual_cam is None:
                    self.virtual_cam = pyvirtualcam.Camera(width=ov_w, height=ov_h, fps=30)
                
                if self.virtual_cam.width != ov_w or self.virtual_cam.height != ov_h:
                     self.virtual_cam.close()
                     self.virtual_cam = pyvirtualcam.Camera(width=ov_w, height=ov_h, fps=30)

                output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
                self.virtual_cam.send(output_rgb)
                self.virtual_cam.sleep_until_next_frame()
            except Exception as e:
                self.vcam_checkbox.setChecked(False)
                self.virtual_cam_active = False
                if self.virtual_cam: self.virtual_cam.close(); self.virtual_cam = None

        # 6. Output to Projector Window
        if self.projector_window:
            # We send the NATIVE projector-space overlay directly
            if self.live_mode:
                self.projector_window.update_frame(proj_overlay)
            elif not self.pattern_checkbox.isChecked():
                # Show black in setup unless grid is shown
                self.projector_window.update_frame(np.zeros_like(proj_overlay))

        # 7. Show in Main UI
        # Add visual mode indicator
        mode_text = "LIVE PERFORMANCE" if self.live_mode else "SETUP MODE (Tracking Muted)"
        color = (0, 0, 255) if self.live_mode else (0, 255, 255)
        cv2.putText(output_canvas, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        qt_img = self.convert_cv_qt(output_canvas)
        self.video_label.setPixmap(qt_img)

    def track_or_detect(self, cv_img):
        """Handles the logic for multiple face tracking and detection."""
        self.frame_count += 1
        
        # 1. Update existing trackers
        new_trackers = []
        for track in self.trackers:
            success, _ = track.update(cv_img)
            # Only keep if tracker succeeds and hasn't failed too long
            if success or track.failed_frames < 5:
                new_trackers.append(track)
        self.trackers = new_trackers

        # 2. Periodically detect new faces
        if self.frame_count % 30 == 0 or len(self.trackers) == 0:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Mask out existing overlays to prevent false positives
            detection_gray = gray.copy()
            for track in self.trackers:
                # Black out the face area
                fx, fy, fw, fh = track.last_box
                cv2.rectangle(detection_gray, (fx, fy), (fx + fw, fy + fh), 0, -1)
                # Black out the overlay area
                if track.last_overlay_bbox:
                    ox1, oy1, ox2, oy2 = track.last_overlay_bbox
                    cv2.rectangle(detection_gray, (ox1, oy1), (ox2, oy2), 0, -1)

            faces = self.face_cascade.detectMultiScale(detection_gray, 1.1, 5)
            
            for face in faces:
                face_box = tuple(map(int, face))
                # Check if this face is already being tracked using IoU
                is_duplicate = False
                for track in self.trackers:
                    if calculate_iou(face_box, track.last_box) > 0.4:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # New face detected
                    self.trackers.append(FaceTrack(face_box, cv_img))
                    self.statusBar().showMessage(f"New face detected. Total: {len(self.trackers)}", 2000)

        # Return all averaged boxes
        return [t.get_avg_box() for t in self.trackers]

    def convert_cv_qt(self, cv_img):
        """Converts an OpenCV image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        p = QPixmap.fromImage(qt_format)
        # Let the QLabel handle the scaling (setScaledContents(True))
        return p


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
