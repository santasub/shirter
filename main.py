import sys
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QCheckBox, QLineEdit,
                             QComboBox, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QFormLayout, QRadioButton,
                             QFileDialog, QStatusBar)


class VideoThread(QThread):
    """A thread that continuously reads frames from a video source."""
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        """Sets a flag to stop the thread's loop."""
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Tracking Overlay Control")
        self.setGeometry(100, 100, 1200, 700)

        # --- State and Processing Setup ---
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.mirror_mode = False
        self.projection_mode = False
        self.offset_x, self.offset_y = 0, 0
        self.tracker = None
        self.face_history = deque(maxlen=5)

        # --- Overlay Properties ---
        self.overlay_mode = 'Text'
        self.overlay_text = "Tracking..."
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.font_thickness = 2
        self.font_color = (255, 255, 255)
        self.overlay_image = None
        self.image_scale = 1.0
        self.font_map = {
            "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
            "Plain": cv2.FONT_HERSHEY_PLAIN,
            "Duplex": cv2.FONT_HERSHEY_DUPLEX,
            "Complex": cv2.FONT_HERSHEY_COMPLEX,
            "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
            "Complex Small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
            "Script Simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            "Script Complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        }

        self.setup_ui()

        # --- Threading Setup ---
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def setup_ui(self):
        """Initializes the user interface."""
        main_layout = QHBoxLayout()
        self.video_label = QLabel("Connecting to camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.video_label, 3)

        control_layout = QVBoxLayout()
        self.setup_core_controls(control_layout)
        self.setup_overlay_type_controls(control_layout)
        self.setup_text_controls(control_layout)
        self.setup_image_controls(control_layout)
        control_layout.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setFixedWidth(320)
        main_layout.addWidget(control_widget, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setStatusBar(QStatusBar(self))

    def setup_core_controls(self, layout):
        core_group = QGroupBox("Core Controls")
        core_layout = QVBoxLayout()
        mirror_checkbox = QCheckBox("Mirror Camera Feed")
        mirror_checkbox.setToolTip("Flip the camera image horizontally.")
        mirror_checkbox.toggled.connect(self.set_mirror_mode)
        core_layout.addWidget(mirror_checkbox)

        projection_checkbox = QCheckBox("Projection Mode")
        tooltip = "Show overlay on a black background for projection."
        projection_checkbox.setToolTip(tooltip)
        projection_checkbox.toggled.connect(self.set_projection_mode)
        core_layout.addWidget(projection_checkbox)

        reset_pos_button = QPushButton("Reset Tracker & Position")
        tooltip = "Reset overlay position and re-initialize tracker."
        reset_pos_button.setToolTip(tooltip)
        reset_pos_button.clicked.connect(self.reset_all)
        core_layout.addWidget(reset_pos_button)

        core_group.setLayout(core_layout)
        layout.addWidget(core_group)

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

        self.font_combo = QComboBox()
        self.font_combo.setToolTip("The font of the overlay text.")
        self.font_combo.addItems(self.font_map.keys())
        self.font_combo.setCurrentText("Simplex")
        self.font_combo.currentIndexChanged.connect(self.set_font_face)
        form_layout.addRow("Font:", self.font_combo)

        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setToolTip("The size of the overlay text.")
        self.scale_spinbox.setRange(0.1, 10.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(self.font_scale)
        self.scale_spinbox.valueChanged.connect(self.set_font_scale)
        form_layout.addRow("Size:", self.scale_spinbox)

        self.thickness_spinbox = QSpinBox()
        self.thickness_spinbox.setToolTip("The thickness of the overlay text.")
        self.thickness_spinbox.setRange(1, 10)
        self.thickness_spinbox.setValue(self.font_thickness)
        self.thickness_spinbox.valueChanged.connect(self.set_font_thickness)
        form_layout.addRow("Thickness:", self.thickness_spinbox)

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

        self.image_group.setLayout(image_form_layout)
        self.image_group.setEnabled(False)
        layout.addWidget(self.image_group)

    # --- Slot methods for controls ---
    def set_mirror_mode(self, checked): self.mirror_mode = checked
    def set_projection_mode(self, checked): self.projection_mode = checked
    def set_overlay_text(self, text): self.overlay_text = text
    def set_font_face(self, i): self.font_face = self.font_map[self.font_combo.itemText(i)]
    def set_font_scale(self, val): self.font_scale = val
    def set_font_thickness(self, val): self.font_thickness = val
    def set_image_scale(self, val): self.image_scale = val

    def reset_all(self):
        self.offset_x, self.offset_y = 0, 0
        self.face_history.clear()
        self.tracker = None
        self.statusBar().showMessage("Tracker and position reset.", 3000)

    def set_overlay_type(self):
        self.overlay_mode = "Image" if self.image_radio.isChecked() else "Text"
        self.text_group.setEnabled(self.overlay_mode == "Text")
        self.image_group.setEnabled(self.overlay_mode == "Image")

    def load_image(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg)")
        if fpath:
            self.overlay_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if self.overlay_image is not None:
                fname = fpath.split('/')[-1]
                self.image_path_label.setText(fname)
                self.statusBar().showMessage(f"Loaded {fname}", 3000)
            else:
                self.image_path_label.setText("Error loading image.")
                self.statusBar().showMessage("Failed to load image.", 3000)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    # --- Drawing and Processing Methods ---
    def draw_overlay(self, canvas, face_box):
        """Draws the selected overlay (text or image) on the canvas."""
        if self.overlay_mode == 'Text':
            self.draw_text_overlay(canvas, face_box)
        elif self.overlay_mode == 'Image' and self.overlay_image is not None:
            self.draw_image_overlay(canvas, face_box)

    def draw_text_overlay(self, canvas, face_box):
        x, y, w, h = face_box
        center_x = x + w // 2
        text_size = cv2.getTextSize(
            self.overlay_text, self.font_face, self.font_scale,
            self.font_thickness)[0]

        base_pos_x = center_x - text_size[0] // 2
        base_pos_y = y - 10
        if base_pos_y < text_size[1]:
            base_pos_y = y + h + text_size[1] + 10

        final_pos = (base_pos_x + self.offset_x, base_pos_y + self.offset_y)
        cv2.putText(canvas, self.overlay_text, final_pos, self.font_face,
                    self.font_scale, self.font_color, self.font_thickness)

    def draw_image_overlay(self, canvas, face_box):
        x, y, w, h = face_box
        center_x, center_y = x + w // 2, y + h // 2

        overlay_h, overlay_w = self.overlay_image.shape[:2]
        scaled_w = int(w * self.image_scale)
        scaled_h = int(overlay_h * (scaled_w / overlay_w))
        resized = cv2.resize(self.overlay_image, (scaled_w, scaled_h))

        pos_x = center_x - scaled_w // 2 + self.offset_x
        pos_y = center_y - scaled_h // 2 + self.offset_y

        # Get the region of interest and handle boundary conditions
        x1, y1 = max(pos_x, 0), max(pos_y, 0)
        x2 = min(pos_x + scaled_w, canvas.shape[1])
        y2 = min(pos_y + scaled_h, canvas.shape[0])
        overlay_x1 = max(0, -pos_x)
        overlay_y1 = max(0, -pos_y)
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        if (x2 > x1) and (y2 > y1):
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

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """The main processing loop for each frame."""
        if self.mirror_mode:
            cv_img = cv2.flip(cv_img, 1)

        output_canvas = np.zeros_like(cv_img) if self.projection_mode else cv_img.copy()
        current_face_box = self.track_or_detect(cv_img)

        if current_face_box:
            self.face_history.append(current_face_box)

        if len(self.face_history) > 0:
            avg_box = tuple(np.mean(self.face_history, axis=0).astype(int))
            self.draw_overlay(output_canvas, avg_box)
            if not self.projection_mode:
                x, y, w, h = avg_box
                cv2.rectangle(
                    output_canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(
                    output_canvas, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)

        qt_img = self.convert_cv_qt(output_canvas)
        self.video_label.setPixmap(qt_img)

    def track_or_detect(self, cv_img):
        """Handles the logic for either tracking or detecting a face."""
        if self.tracker:
            success, box = self.tracker.update(cv_img)
            if success:
                return tuple(map(int, box))
            else:
                self.tracker = None
                self.face_history.clear()
                self.statusBar().showMessage(
                    "Tracker lost. Re-detecting...", 2000)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            best_face = sorted(
                faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(cv_img, tuple(best_face))
            self.statusBar().showMessage("Tracker initialized.", 2000)
            return tuple(best_face)
        return None

    def convert_cv_qt(self, cv_img):
        """Converts an OpenCV image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(qt_format)
        scaled_pixmap = p.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio)
        return scaled_pixmap


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
