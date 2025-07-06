import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("DJ Shirt Visualizer")

        self.cap = None
        self.logo_image_original = None # Initialize attribute

        # --- Tracking variables ---
        self.tracker = None
        self.tracker_initialized = False
        self.last_known_bbox = None # Stores (x,y,w,h) of the last known good position
        self.current_bbox_for_warp = None # Bbox to use for current frame's warp

        # --- Video Loop & Processing Flags ---
        self.camera_active = False  # Flag to control the unified video loop
        self.after_id_video_loop = None  # To store ID of root.after() call
        self.is_processing = False # Main flag to enable/disable shirt processing logic

        # HSV Color range defaults (e.g., for a shade of green)
        # User will be able to adjust this.
        self.lower_hsv = np.array([35, 100, 100]) # Increased min S and V
        self.upper_hsv = np.array([85, 255, 255])
        self.hsv_color_display = None # For showing selected color swatch

        # --- Main PanedWindow Setup ---
        self.main_paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Video Frame (Left Pane) ---
        self.video_frame = ttk.Frame(self.main_paned_window, padding="5")
        # self.video_frame.pack(fill=tk.BOTH, expand=True) # pack/grid is handled by add
        self.main_paned_window.add(self.video_frame, stretch="always", minsize=300) # stretch="always"

        # --- Control Frame (Right Pane) ---
        # This is the main container for controls, will be added to the PanedWindow
        self.control_panel_outer_frame = ttk.Frame(self.main_paned_window, padding="5")
        self.main_paned_window.add(self.control_panel_outer_frame, stretch="never", minsize=300) # Increased minsize a bit

        # Configure grid for the control_panel_outer_frame to hold Notebook and Start/Stop button
        self.control_panel_outer_frame.grid_rowconfigure(0, weight=1) # Notebook will expand
        self.control_panel_outer_frame.grid_rowconfigure(1, weight=0) # Button fixed at bottom
        self.control_panel_outer_frame.grid_columnconfigure(0, weight=1)

        # Create the Notebook (Tabs)
        self.notebook = ttk.Notebook(self.control_panel_outer_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Create frames for each tab
        self.tab_camera_color = ttk.Frame(self.notebook, padding="5")
        self.tab_overlays = ttk.Frame(self.notebook, padding="5")

        self.notebook.add(self.tab_camera_color, text='Camera & Color')
        self.notebook.add(self.tab_overlays, text='Overlays')

        # --- Video Display Area (within self.video_frame) ---
        # Set a default size for video labels to prevent collapsing
        self.default_video_bg = Image.new('RGB', (640, 480), (100, 100, 100))
        self.default_video_img = ImageTk.PhotoImage(self.default_video_bg)

        self.raw_video_label = ttk.Label(self.video_frame, image=self.default_video_img)
        self.raw_video_label.pack(pady=5, padx=5, expand=True, fill="both") # fill and expand within video_frame

        self.processed_video_label = ttk.Label(self.video_frame, image=self.default_video_img)
        self.processed_video_label.pack(pady=5, padx=5, expand=True, fill="both") # fill and expand within video_frame

        # Store current dimensions of video labels for resizing
        self.raw_video_label_width = self.default_video_bg.width
        self.raw_video_label_height = self.default_video_bg.height
        self.processed_video_label_width = self.default_video_bg.width
        self.processed_video_label_height = self.default_video_bg.height

        self.raw_video_label.bind("<Configure>", self._on_raw_video_resize)
        self.processed_video_label.bind("<Configure>", self._on_processed_video_resize)

        # --- Populate Tabs ---
        self._populate_camera_color_tab(self.tab_camera_color)
        self._populate_overlays_tab(self.tab_overlays)

        # Start/Stop Button (Now part of control_panel_outer_frame, below the notebook)
        self.start_stop_button = ttk.Button(self.control_panel_outer_frame, text="Start Processing", command=self.toggle_processing)
        self.start_stop_button.grid(row=1, column=0, sticky="ew", padx=5, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.camera_active = False  # Flag to control the unified video loop
        self.after_id_video_loop = None  # To store ID of root.after() call
        self.is_processing = False # Ensure this is initialized before _initial_camera_start might use it implicitly

        # Attempt to start initial camera preview
        self._initial_camera_start()


    # Removed _bind_mousewheel as the canvas and scrollbar are removed.
    # If Notebook content overflows a tab, Tkinter handles it or we might need per-tab scrollable frames later.

    def _initial_camera_start(self):
        if self.available_cameras and self.initial_cam_name != "No cameras found":
            print(f"DEBUG: Attempting initial camera start with: {self.initial_cam_name}")
            self._start_camera_feed(camera_name=self.initial_cam_name)
        else:
            print("DEBUG: No cameras available for initial preview or 'No cameras found' selected.")
            # Ensure UI reflects that no camera is active
            self.raw_video_label.configure(image=self.default_video_img)
            self.raw_video_label.image = self.default_video_img
            self.processed_video_label.configure(image=self.default_video_img)
            self.processed_video_label.image = self.default_video_img
            self.camera_dropdown.config(state=tk.DISABLED)


    def _start_camera_feed(self, camera_name=None):
        if self.after_id_video_loop:
            self.root.after_cancel(self.after_id_video_loop)
            self.after_id_video_loop = None

        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
                print("DEBUG: Previous camera released in _start_camera_feed.")
            except Exception as e:
                print(f"DEBUG: Exception releasing previous camera in _start_camera_feed: {e}")
        self.cap = None
        self.camera_active = False

        if camera_name is None: # If called from toggle_processing without a specific camera
            camera_name = self.camera_var.get()

        if not camera_name or camera_name == "No cameras found":
            messagebox.showerror("Error", "No camera selected or available.")
            self.raw_video_label.configure(image=self.default_video_img) # Reset preview
            self.raw_video_label.image = self.default_video_img
            self.processed_video_label.configure(image=self.default_video_img)
            self.processed_video_label.image = self.default_video_img
            self.camera_dropdown.config(state=tk.DISABLED if not self.available_cameras else tk.NORMAL)
            self.start_stop_button.config(text="Start Processing")
            self.is_processing = False # Ensure processing is marked as off
            return False # Indicate failure

        cam_idx = self.available_cameras.get(camera_name, -1)
        if cam_idx == -1:
            messagebox.showerror("Error", f"Invalid camera selected: {camera_name}")
            return False

        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_idx, backend)

        if not self.cap.isOpened():
            error_message = f"Could not open {camera_name} (Index {cam_idx})."
            if sys.platform == 'darwin':
                error_message += "\nPlease ensure app has camera permission (System Settings > Privacy & Security > Camera)."
            messagebox.showerror("Error", error_message)
            self.cap = None
            self.camera_active = False
            self.raw_video_label.configure(image=self.default_video_img) # Reset preview
            self.raw_video_label.image = self.default_video_img
            return False

        print(f"DEBUG: Camera {camera_name} (Index {cam_idx}) opened successfully.")
        self.camera_active = True
        self.camera_dropdown.config(state=tk.DISABLED if self.is_processing else tk.NORMAL) # Disable during processing
        self._update_video_feeds_loop() # Start the unified loop
        return True


    def _on_camera_select(self, selected_camera_name):
        """Called when a new camera is selected from the OptionMenu."""
        print(f"DEBUG: Camera selected via OptionMenu: {selected_camera_name}")
        if not self.is_processing: # Only switch preview if not actively processing
            self._start_camera_feed(camera_name=selected_camera_name)
        else:
            # If processing, user needs to stop and restart to change camera.
            # Or, we could allow dynamic switching, but that's more complex.
            print("INFO: To change camera, please stop processing first.")
            # Revert OptionMenu to the currently active camera if different
            # This requires knowing which camera self.cap is using. For now, just inform.
            # self.camera_var.set(current_active_camera_name) # TODO if needed


    def _populate_camera_color_tab(self, tab_frame):
        # Camera Selection
        camera_select_frame = ttk.LabelFrame(tab_frame, text="Camera Setup")
        camera_select_frame.pack(fill="x", expand=False, padx=5, pady=5)

        self.camera_var = tk.StringVar()
        self.available_cameras = self._list_available_cameras()
        print(f"DEBUG: self.available_cameras after call: {self.available_cameras}")

        camera_names = list(self.available_cameras.keys())
        self.initial_cam_name = "No cameras found" # Made instance variable for _initial_camera_start
        cam_dropdown_state = tk.DISABLED

        if camera_names:
            self.initial_cam_name = camera_names[0]
            cam_dropdown_state = tk.NORMAL
        else:
            camera_names = [self.initial_cam_name]
            if self.initial_cam_name not in self.available_cameras: # Should be caught by _list_available_cameras
                 self.available_cameras[self.initial_cam_name] = -1 # Ensure it exists if no cams found

        self.camera_var.set(self.initial_cam_name)
        # Add command to OptionMenu to handle camera changes
        self.camera_dropdown = tk.OptionMenu(camera_select_frame, self.camera_var, *camera_names, command=self._on_camera_select)
        self.camera_dropdown.pack(pady=5, padx=5, fill="x")
        self.camera_dropdown.config(state=cam_dropdown_state)
        # print is now part of _list_available_cameras or _start_camera_feed

        # Color Selection (HSV)
        color_frame = ttk.LabelFrame(tab_frame, text="Shirt Color (HSV)")
        color_frame.pack(fill="x", expand=False, padx=5, pady=5)

        self.pick_color_button = ttk.Button(color_frame, text="Pick Shirt Color (from Preview)", command=self.enable_color_picker_mode)
        self.pick_color_button.pack(pady=2)
        self.color_picker_mode = False

        self.hsv_color_display_label = ttk.Label(color_frame, text="Selected Color:")
        self.hsv_color_display_label.pack(pady=2)
        self.hsv_color_swatch = tk.Label(color_frame, background="grey", width=10, height=2)
        self.hsv_color_swatch.pack(pady=2)
        self._update_color_swatch(self.lower_hsv, self.upper_hsv)

        ttk.Label(color_frame, text="Picked Color Tolerances:").pack(pady=(5,0))
        self.hue_tolerance_var = tk.IntVar(value=10)
        self.base_h_picked = -1
        hue_tolerance_slider_frame = ttk.Frame(color_frame)
        hue_tolerance_slider_frame.pack(fill="x", expand=True)
        ttk.Label(hue_tolerance_slider_frame, text="H Tol:").pack(side=tk.LEFT, padx=2)
        self.hue_tolerance_slider = tk.Scale(hue_tolerance_slider_frame, from_=0, to=30, orient=tk.HORIZONTAL, variable=self.hue_tolerance_var, command=self._update_hsv_from_tolerance_sliders, length=150)
        self.hue_tolerance_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        self.s_min_var = tk.IntVar(value=50)
        self.s_max_var = tk.IntVar(value=255)
        self.base_s_picked = -1
        s_range_frame = ttk.Frame(color_frame)
        s_range_frame.pack(fill="x", expand=True, pady=2)
        ttk.Label(s_range_frame, text="S Min:").pack(side=tk.LEFT, padx=2)
        self.s_min_slider = tk.Scale(s_range_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.s_min_var, command=self._update_hsv_from_tolerance_sliders, length=150)
        self.s_min_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=2)
        ttk.Label(s_range_frame, text="S Max:").pack(side=tk.LEFT, padx=2)
        self.s_max_slider = tk.Scale(s_range_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.s_max_var, command=self._update_hsv_from_tolerance_sliders, length=150)
        self.s_max_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        self.v_min_var = tk.IntVar(value=50)
        self.v_max_var = tk.IntVar(value=255)
        self.base_v_picked = -1
        v_range_frame = ttk.Frame(color_frame)
        v_range_frame.pack(fill="x", expand=True, pady=2)
        ttk.Label(v_range_frame, text="V Min:").pack(side=tk.LEFT, padx=2)
        self.v_min_slider = tk.Scale(v_range_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.v_min_var, command=self._update_hsv_from_tolerance_sliders, length=150)
        self.v_min_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=2)
        ttk.Label(v_range_frame, text="V Max:").pack(side=tk.LEFT, padx=2)
        self.v_max_slider = tk.Scale(v_range_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.v_max_var, command=self._update_hsv_from_tolerance_sliders, length=150)
        self.v_max_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=2)

        ttk.Label(color_frame, text="Resulting HSV Range:").pack(pady=(5,0))
        hsv_sliders_frame = ttk.Frame(color_frame)
        hsv_sliders_frame.pack(fill="x", expand=True)
        self.hsv_sliders = {}
        for i, label in enumerate(["H_low", "S_low", "V_low", "H_high", "S_high", "V_high"]):
            val = self.lower_hsv[i % 3] if "low" in label else self.upper_hsv[i % 3]
            max_val = 179 if "H_" in label else 255
            scale = tk.Scale(hsv_sliders_frame, from_=0, to=max_val, orient=tk.HORIZONTAL, label=label, length=200, command=self._update_hsv_from_main_sliders)
            scale.set(val)
            scale.pack(fill="x", padx=2)
            self.hsv_sliders[label] = scale

        if self.base_h_picked == -1: self.base_h_picked = (self.lower_hsv[0] + self.upper_hsv[0]) // 2
        if self.base_s_picked == -1: self.base_s_picked = 128
        if self.base_v_picked == -1: self.base_v_picked = 128
        self._update_hsv_from_tolerance_sliders()

    def _populate_overlays_tab(self, tab_frame):
        # Text Overlay
        text_overlay_frame = ttk.LabelFrame(tab_frame, text="Text Overlay")
        text_overlay_frame.pack(fill="x", expand=False, padx=5, pady=5)
        self.text_entry_var = tk.StringVar(value="DJ SHIRT")
        self.text_entry = ttk.Entry(text_overlay_frame, textvariable=self.text_entry_var)
        self.text_entry.pack(pady=5, padx=5, fill="x")

        # Logo Overlay
        logo_overlay_frame = ttk.LabelFrame(tab_frame, text="Logo Overlay")
        logo_overlay_frame.pack(fill="x", expand=False, padx=5, pady=5)
        ttk.Button(logo_overlay_frame, text="Upload Logo", command=self.upload_logo).pack(pady=2)
        self.logo_path_var = tk.StringVar(value="No logo selected.")
        ttk.Label(logo_overlay_frame, textvariable=self.logo_path_var, wraplength=180).pack(pady=2, padx=5, fill="x")

        # ROI Definition (Placeholder)
        roi_frame = ttk.LabelFrame(tab_frame, text="Overlay Regions (ROI)")
        roi_frame.pack(fill="x", expand=False, padx=5, pady=5)
        ttk.Button(roi_frame, text="Define Shirt Regions", command=self.define_roi, state=tk.DISABLED).pack(pady=5)


    def _on_raw_video_resize(self, event):
        self.raw_video_label_width = event.width
        self.raw_video_label_height = event.height

    def _on_processed_video_resize(self, event):
        self.processed_video_label_width = event.width
        self.processed_video_label_height = event.height

    def _resize_frame_keep_aspect_ratio(self, frame, target_width, target_height):
        if target_width <= 0 or target_height <= 0: # Avoid division by zero or invalid sizes
            return frame

        original_height, original_width = frame.shape[:2]
        if original_width == 0 or original_height == 0:
            return frame # Invalid original frame

        # Calculate aspect ratios
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            # Original is wider than target, so fit to width
            new_width = target_width
            new_height = int(new_width / original_aspect)
        else:
            # Original is taller than target (or same aspect), so fit to height
            new_height = target_height
            new_width = int(new_height * original_aspect)

        if new_width <=0 or new_height <=0: # if calculation results in zero for a dim
            return frame

        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _list_available_cameras(self):
        """Lists available camera devices and their indices."""
        cameras = {}
        index = 0
        # Try up to a certain number of indices to avoid infinite loops on some systems
        max_cam_tests = 10
        while index < max_cam_tests:
            # For Windows, CAP_DSHOW. For macOS, CAP_AVFOUNDATION. For Linux, default or CAP_V4L2.
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
            cap_test = cv2.VideoCapture(index, backend)
            if cap_test.isOpened():
                cameras[f"Camera {index}"] = index
                cap_test.release()
            else:
                # For some systems, if a higher index fails, lower ones might not exist either.
                # However, it's also possible for indices to be non-contiguous (e.g., 0, 2).
                # So we continue checking up to max_cam_tests.
                # If the first few fail, it's unlikely many more will succeed.
                if index > 3 and not cameras: # Heuristic: if first 4 attempts fail, probably no cams
                    break
                cap_test.release() # Ensure it's released even if not opened
            index += 1

        if not cameras and index == max_cam_tests : # If loop finished due to max_cam_tests and no cameras found
            # This case might indicate a broader issue like no backend support or global permission denial
            print("No cameras found after checking multiple indices. Ensure camera drivers and permissions are correct.")
        elif not cameras: # If loop broke early or just no cameras
            cameras[f"Camera {index}"] = index
            cap_test.release()
            index += 1
        if not cameras:
            print("No cameras found!")
        else:
            print("Available cameras:", cameras)
        return cameras

    def _update_hsv_from_main_sliders(self, event=None): # Renamed from _update_hsv_from_sliders
        self.lower_hsv[0] = self.hsv_sliders["H_low"].get()
        self.lower_hsv[1] = self.hsv_sliders["S_low"].get()
        self.lower_hsv[2] = self.hsv_sliders["V_low"].get()
        self.upper_hsv[0] = self.hsv_sliders["H_high"].get()
        self.upper_hsv[1] = self.hsv_sliders["S_high"].get()
        self.upper_hsv[2] = self.hsv_sliders["V_high"].get()
        self._update_color_swatch(self.lower_hsv, self.upper_hsv)
        # When main sliders are moved, it implies direct manipulation, potentially overriding the "picked + tolerance" logic.
        # We might need to update base_h_picked etc. here, or provide a button to "reset to picked + tolerance"
        # For now, this direct override is fine.

    def _update_hsv_from_tolerance_sliders(self, event=None):
        if self.base_h_picked == -1: # No color picked yet, or reset
            # If no color is picked, the tolerance sliders might not have a meaningful base.
            # Option 1: Disable tolerance sliders until a color is picked.
            # Option 2: Let them adjust the current H_low/H_high, S_low/S_high etc. directly,
            #           which is effectively what _update_hsv_from_main_sliders does.
            # For now, let's assume they adjust the current self.lower_hsv/upper_hsv if no base is picked,
            # or we ensure a base is picked/defaulted.
            # The current initial _update_hsv_from_tolerance_sliders call in __init__ will set a default range.
            # Let's ensure base_h_picked is set from current lower/upper if it's -1
            # This makes the tolerance sliders always work relative to *some* base H.
            current_h_low = self.hsv_sliders["H_low"].get()
            current_h_high = self.hsv_sliders["H_high"].get()
            self.base_h_picked = (current_h_low + current_h_high) // 2
            # S and V min/max are directly set by their sliders, no base_s/v needed for calculation from tolerance.

        h_tolerance = self.hue_tolerance_var.get()

        # Calculate Hue range based on base_h_picked and tolerance
        # Handle Hue wrap-around (0-179 degrees)
        # A simple way:
        h_low = self.base_h_picked - h_tolerance
        h_high = self.base_h_picked + h_tolerance

        if h_low < 0: h_low = 0 # Clamp, or handle wrap-around for more complex scenarios
        if h_high > 179: h_high = 179 # Clamp

        # For more robust hue that wraps (e.g. if base_H is 5, tol is 10, range is 175-15):
        # This requires splitting into two ranges if it wraps, which cv2.inRange handles if called twice.
        # For simplicity with single range sliders, we'll use simple clamping for now.
        # If we wanted true hue wrapping, the UI would need to show two ranges or the mask logic change.

        self.lower_hsv[0] = h_low
        self.upper_hsv[0] = h_high

        self.lower_hsv[1] = self.s_min_var.get()
        self.upper_hsv[1] = self.s_max_var.get()
        self.lower_hsv[2] = self.v_min_var.get()
        self.upper_hsv[2] = self.v_max_var.get()

        # Ensure min <= max for S and V
        if self.lower_hsv[1] > self.upper_hsv[1]: self.lower_hsv[1] = self.upper_hsv[1]
        if self.lower_hsv[2] > self.upper_hsv[2]: self.lower_hsv[2] = self.upper_hsv[2]

        # Update the 6 main display sliders
        self.hsv_sliders["H_low"].set(self.lower_hsv[0])
        self.hsv_sliders["S_low"].set(self.lower_hsv[1])
        self.hsv_sliders["V_low"].set(self.lower_hsv[2])
        self.hsv_sliders["H_high"].set(self.upper_hsv[0])
        self.hsv_sliders["S_high"].set(self.upper_hsv[1])
        self.hsv_sliders["V_high"].set(self.upper_hsv[2])

        self._update_color_swatch(self.lower_hsv, self.upper_hsv)


    def _update_color_swatch(self, lower_hsv, upper_hsv):
        # Create an average color for the swatch
        # Ensure lower <= upper for sensible average calculation, though GUI should enforce this.
        avg_h = int((max(0,lower_hsv[0]) + min(179,upper_hsv[0])) / 2)
        avg_s = int((lower_hsv[1] + upper_hsv[1]) / 2)
        avg_v = int((lower_hsv[2] + upper_hsv[2]) / 2)

        # Convert this average HSV to RGB for display
        avg_hsv_color = np.uint8([[[avg_h, avg_s, avg_v]]])
        rgb_color = cv2.cvtColor(avg_hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        hex_color = f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'
        self.hsv_color_swatch.config(background=hex_color)

    def enable_color_picker_mode(self):
        if not self.is_processing:
            messagebox.showwarning("Camera Off", "Please start the camera feed first to pick a color.")
            return
        self.color_picker_mode = True
        self.pick_color_button.config(text="Click on Shirt in Raw Feed", state=tk.DISABLED)
        self.raw_video_label.bind("<Button-1>", self._on_raw_video_click)
        print("Color picker mode enabled. Click on the raw video feed.")

    def _on_raw_video_click(self, event):
        if self.color_picker_mode and hasattr(self, 'current_raw_frame_for_picker'):
            # Coordinates are relative to the label; need to scale if image is resized.
            # For simplicity, assume label displays image at original size for now.
            # TODO: Handle resized image coordinates if raw_video_label resizes the image.
            x, y = event.x, event.y

            # Ensure coordinates are within frame bounds
            height, width, _ = self.current_raw_frame_for_picker.shape
            if 0 <= y < height and 0 <= x < width:
                bgr_color = self.current_raw_frame_for_picker[y, x]
                hsv_color_numpy = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

                # Convert numpy.uint8 to standard Python int before arithmetic
                h_picked = int(hsv_color_numpy[0])
                s_picked = int(hsv_color_numpy[1])
                v_picked = int(hsv_color_numpy[2])
                print(f"Clicked BGR: {bgr_color}, Picked HSV: ({h_picked}, {s_picked}, {v_picked})")

                # --- Store the picked base color ---
                self.base_h_picked = h_picked
                self.base_s_picked = s_picked
                self.base_v_picked = v_picked

                # --- Update the new tolerance/range sliders based on the picked color ---
                # (Assuming these sliders and their variables like self.hue_tolerance_var exist)
                # For now, I'll just print what would be set. Actual slider creation is next.

                # Example: Set default tolerances/ranges after picking
                # These values would ideally come from default settings for the new sliders
                default_hue_tolerance = 10
                default_s_min = max(0, s_picked - 75)
                default_s_max = min(255, s_picked + 75)
                default_v_min = max(0, v_picked - 75)
                default_v_max = min(255, v_picked + 75)

                if hasattr(self, 'hue_tolerance_slider'): # Check if new sliders are implemented
                    self.hue_tolerance_var.set(default_hue_tolerance)
                    self.s_min_var.set(default_s_min)
                    self.s_max_var.set(default_s_max)
                    self.v_min_var.set(default_v_min)
                    self.v_max_var.set(default_v_max)
                    # This will trigger the calculation and update of main sliders
                    self._update_hsv_from_tolerance_sliders()
                # Removed the 'else' block that had the direct old calculation,
                # as the new sliders are now considered implemented.

                print(f"DEBUG: Base HSV set to: H={self.base_h_picked}, S={self.base_s_picked}, V={self.base_v_picked}")
                print(f"DEBUG: Tolerance/Range sliders set to: H_tol={self.hue_tolerance_var.get()}, S_range=({self.s_min_var.get()}-{self.s_max_var.get()}), V_range=({self.v_min_var.get()}-{self.v_max_var.get()})")

            self.color_picker_mode = False
            self.pick_color_button.config(text="Pick Shirt Color (from Preview)", state=tk.NORMAL)
            self.raw_video_label.unbind("<Button-1>")


    def upload_logo(self):
        filepath = filedialog.askopenfilename(
            title="Select Logo Image",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        if filepath:
            try:
                print(f"DEBUG: Attempting to load logo from: {filepath}")
                # Load the image using Pillow
                img = Image.open(filepath)
                img.load() # Force loading the image data

                # Store the original image. We might need to resize/process it later for display or overlay.
                self.logo_image_original = img

                # For now, just confirm it's loaded.
                # We could display a small thumbnail in the UI if desired later.
                self.logo_path_var.set(os.path.basename(filepath)) # Show filename
                print(f"Successfully loaded logo: {filepath}, format: {img.format}, size: {img.size}, mode: {img.mode}")
                messagebox.showinfo("Logo Loaded", f"Successfully loaded logo: {os.path.basename(filepath)}")

                # TODO: Trigger a re-render or update if video is processing, to show the logo
                # This part will be handled when we implement logo rendering on the shirt.

            except FileNotFoundError:
                messagebox.showerror("Error", f"Logo file not found: {filepath}")
                self.logo_path_var.set("Error: File not found.")
                print(f"ERROR: Logo file not found: {filepath}")
            except UnidentifiedImageError:
                messagebox.showerror("Error", f"Cannot identify image file. Is it a valid image format (PNG, JPG, etc.)?\nFile: {filepath}")
                self.logo_path_var.set("Error: Invalid image format.")
                print(f"ERROR: Cannot identify image file: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading the logo: {e}")
                self.logo_path_var.set("Error: Could not load logo.")
                print(f"ERROR: Could not load logo '{filepath}': {e}")


    def define_roi(self):
        # Placeholder for ROI definition logic
        print("ROI definition button clicked. To be implemented.")
        # This will likely involve capturing a frame and drawing on it.

    def toggle_processing(self):
        self.is_processing = not self.is_processing
        if self.is_processing:
            cam_idx_str = self.camera_var.get()
            if not self.camera_active: # If camera isn't already running (e.g. from initial preview)
                if not self._start_camera_feed(): # Try to start it with current selection
                    self.is_processing = False # Ensure processing flag is off if camera failed
                    return # _start_camera_feed shows messages

            # If camera is active (either from preview or just started)
            self.is_processing = True
            self.start_stop_button.config(text="Stop Processing")
            self.camera_dropdown.config(state=tk.DISABLED) # Disable camera selection while processing
            self.pick_color_button.config(state=tk.NORMAL)
            print("Processing Started")
            # The _update_video_feeds_loop is already running if camera_active is true,
            # it will pick up the is_processing flag.
        else: # Stopping processing
            self.is_processing = False
            self.start_stop_button.config(text="Start Processing")
            self.camera_dropdown.config(state=tk.NORMAL if self.available_cameras and self.initial_cam_name != "No cameras found" else tk.DISABLED)
            self.pick_color_button.config(state="disabled")

            if self.color_picker_mode:
                self.color_picker_mode = False
                self.raw_video_label.unbind("<Button-1>")
            print("Processing Stopped")

            # Reset processed video label to default, raw preview continues via _update_video_feeds_loop
            self.processed_video_label.configure(image=self.default_video_img)
            self.processed_video_label.image = self.default_video_img


    def _update_video_feeds_loop(self): # Renamed from _video_loop
        if not self.camera_active or not self.cap or not self.cap.isOpened():
            # If camera becomes inactive unexpectedly, stop the loop and reset UI
            print("DEBUG: Camera feed loop stopping as camera is not active or not opened.")
            self.camera_active = False
            self.is_processing = False # Also stop processing
            self.raw_video_label.configure(image=self.default_video_img)
            self.raw_video_label.image = self.default_video_img
            self.processed_video_label.configure(image=self.default_video_img)
            self.processed_video_label.image = self.default_video_img
            self.start_stop_button.config(text="Start Processing")
            is_cam_available = bool(self.available_cameras and self.initial_cam_name != "No cameras found")
            self.camera_dropdown.config(state=tk.NORMAL if is_cam_available else tk.DISABLED)
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?).")
            self.after_id_video_loop = self.root.after(100, self._update_video_feeds_loop) # Try again shortly
            return

        # Always display raw frame
        self.current_raw_frame_for_picker = frame.copy() # For color picker

        target_raw_w = self.raw_video_label_width - 10 if self.raw_video_label_width > 20 else self.raw_video_label_width
        target_raw_h = self.raw_video_label_height - 10 if self.raw_video_label_height > 20 else self.raw_video_label_height
        resized_raw_frame = self._resize_frame_keep_aspect_ratio(frame, target_raw_w, target_raw_h)
        img_raw = cv2.cvtColor(resized_raw_frame, cv2.COLOR_BGR2RGB)
        img_raw = Image.fromarray(img_raw)
        imgtk_raw = ImageTk.PhotoImage(image=img_raw)
        self.raw_video_label.imgtk = imgtk_raw
        self.raw_video_label.configure(image=imgtk_raw)

        if self.is_processing:
            # --- Start Processing Logic ---
            processed_frame_display = frame.copy() # Start with a fresh copy of the original frame for processing

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            current_lower_hsv = self.lower_hsv
            current_upper_hsv = self.upper_hsv
            mask = cv2.inRange(hsv_frame, current_lower_hsv, current_upper_hsv)

            self.current_bbox_for_warp = None # Reset for current frame

            if self.tracker_initialized:
                success, bbox = self.tracker.update(frame)
                if success:
                    self.last_known_bbox = bbox
                    self.current_bbox_for_warp = bbox
                    # (x, y, w, h) = [int(v) for v in bbox]
                    # cv2.rectangle(processed_frame_display, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue for tracker
                else:
                    print("Tracker lost object.")
                    self.tracker_initialized = False
                    self.tracker = None # Destroy tracker
                    # Will attempt re-detection via color below

            if not self.tracker_initialized:
                # Optional: Morphological operations
                # kernel = np.ones((5,5), np.uint8)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    min_contour_area = 500 # TODO: Make configurable
                    if cv2.contourArea(largest_contour) > min_contour_area:
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        self.current_bbox_for_warp = (x,y,w,h)
                        self.last_known_bbox = (x,y,w,h)

                        # Initialize tracker
                        try:
                            # self.tracker = cv2.TrackerKCF_create() # Faster, less accurate
                            self.tracker = cv2.TrackerCSRT_create() # Slower, more accurate
                            self.tracker.init(frame, self.last_known_bbox)
                            self.tracker_initialized = True
                            print(f"Tracker initialized with bbox: {self.last_known_bbox}")
                        except Exception as e:
                            print(f"Error initializing tracker: {e}")
                            self.tracker = None
                            self.tracker_initialized = False
                        # cv2.rectangle(processed_frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green for detection

            # Perspective Warp Logic (uses self.current_bbox_for_warp)
            if self.current_bbox_for_warp:
                x, y, w, h = [int(v) for v in self.current_bbox_for_warp]
                # Define target points as corners of the bounding box for warp
                # This is a simplification; ideally, we'd get a better quad from contour or tracker if possible
                # For CSRT/KCF, they give a bbox. We can use approxPolyDP on a mask from this bbox if needed.
                # For now, using the bbox directly as the target quad.
                temp_quad_points = np.array([
                    [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                ], dtype=np.float32)

                # Order points: top-left, top-right, bottom-right, bottom-left
                rect = np.zeros((4, 2), dtype="float32")
                s = temp_quad_points.sum(axis=1)
                rect[0] = temp_quad_points[np.argmin(s)]
                rect[2] = temp_quad_points[np.argmax(s)]
                diff = np.diff(temp_quad_points, axis=1)
                rect[1] = temp_quad_points[np.argmin(diff)]
                rect[3] = temp_quad_points[np.argmax(diff)]
                ordered_target_points = rect

                # Text Overlay (Warped)
                text_to_overlay = self.text_entry_var.get()
                if not text_to_overlay.strip(): text_to_overlay = " "
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_color_bgr = (0, 0, 0)
                bg_color_bgr = (255, 255, 255)
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(text_to_overlay, font_face, font_scale, thickness)
                padding = 20
                src_w = text_w + 2 * padding
                src_h = text_h + baseline + 2 * padding
                text_texture = np.full((src_h, src_w, 3), bg_color_bgr, dtype=np.uint8)
                text_origin_x = padding
                text_origin_y = text_h + padding
                cv2.putText(text_texture, text_to_overlay, (text_origin_x, text_origin_y), font_face, font_scale, font_color_bgr, thickness, cv2.LINE_AA)
                source_points = np.array([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], dtype=np.float32)

                if ordered_target_points.shape == (4,2) and source_points.shape == (4,2): # Check if points are valid
                    matrix = cv2.getPerspectiveTransform(source_points, ordered_target_points)
                    warped_text_texture = cv2.warpPerspective(text_texture, matrix, (processed_frame_display.shape[1], processed_frame_display.shape[0]))
                    mask_for_blending = cv2.inRange(warped_text_texture, np.array([0,0,0]), np.array([250,250,250]))
                    mask_for_blending = cv2.bitwise_not(mask_for_blending)
                    processed_frame_display = cv2.bitwise_and(processed_frame_display, processed_frame_display, mask=cv2.bitwise_not(mask_for_blending))
                    processed_frame_display = cv2.bitwise_or(processed_frame_display, warped_text_texture, mask=mask_for_blending)
                else:
                    print("DEBUG: Invalid points for perspective transform.")
            # --- End Processing Logic ---

            # Display processed frame
            target_processed_w = self.processed_video_label_width - 10 if self.processed_video_label_width > 20 else self.processed_video_label_width
            target_processed_h = self.processed_video_label_height - 10 if self.processed_video_label_height > 20 else self.processed_video_label_height
            resized_processed_frame = self._resize_frame_keep_aspect_ratio(processed_frame_display, target_processed_w, target_processed_h)
            img_processed_tk = cv2.cvtColor(resized_processed_frame, cv2.COLOR_BGR2RGB) # Renamed to avoid conflict
            img_processed_pil = Image.fromarray(img_processed_tk) # Renamed
            imgtk_processed = ImageTk.PhotoImage(image=img_processed_pil) # Renamed
            self.processed_video_label.imgtk = imgtk_processed
            self.processed_video_label.configure(image=imgtk_processed)
        else:
            # If not processing, show default image on processed_video_label
            self.processed_video_label.configure(image=self.default_video_img)
            self.processed_video_label.image = self.default_video_img


        # Schedule next frame update
        if self.camera_active: # Check if we should continue the loop
            self.after_id_video_loop = self.root.after(15, self._update_video_feeds_loop)

    def on_closing(self):
        print("DEBUG: Closing application...")
        self.camera_active = False # Signal loop to stop
        self.is_processing = False
        if self.after_id_video_loop:
            self.root.after_cancel(self.after_id_video_loop)
            self.after_id_video_loop = None
            print("DEBUG: Video loop canceled via after_cancel.")

        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
                print("DEBUG: Camera released in on_closing.")
            except Exception as e:
                print(f"DEBUG: Exception during camera release in on_closing: {e}")
            finally:
                self.cap = None
        self.root.destroy()

    # def run(self): # No longer needed, mainloop is called in __main__
        # Subtract a small amount for padding within the label, if any
        target_raw_w = self.raw_video_label_width - 10 if self.raw_video_label_width > 20 else self.raw_video_label_width
        target_raw_h = self.raw_video_label_height - 10 if self.raw_video_label_height > 20 else self.raw_video_label_height

        resized_raw_frame = self._resize_frame_keep_aspect_ratio(frame, target_raw_w, target_raw_h)
        img_raw = cv2.cvtColor(resized_raw_frame, cv2.COLOR_BGR2RGB)
        img_raw = Image.fromarray(img_raw)
        imgtk_raw = ImageTk.PhotoImage(image=img_raw)
        self.raw_video_label.imgtk = imgtk_raw
        self.raw_video_label.configure(image=imgtk_raw)

        # Resize processed frame (which is processed_frame_display)
        target_processed_w = self.processed_video_label_width - 10 if self.processed_video_label_width > 20 else self.processed_video_label_width
        target_processed_h = self.processed_video_label_height - 10 if self.processed_video_label_height > 20 else self.processed_video_label_height

        resized_processed_frame = self._resize_frame_keep_aspect_ratio(processed_frame_display, target_processed_w, target_processed_h)
        img_processed = cv2.cvtColor(resized_processed_frame, cv2.COLOR_BGR2RGB)
        img_processed = Image.fromarray(img_processed)
        imgtk_processed = ImageTk.PhotoImage(image=img_processed)
        self.processed_video_label.imgtk = imgtk_processed
        self.processed_video_label.configure(image=imgtk_processed)

        if self.is_processing:
            self.root.after(15, self._video_loop) # Approx 66 FPS, adjust as needed for performance

    def on_closing(self):
        if self.is_processing:
            self.is_processing = False # Stop the loop
        if self.cap:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    print("DEBUG: Camera released in on_closing.")
            except Exception as e:
                print(f"DEBUG: Exception during camera release in on_closing: {e}")
            finally:
                self.cap = None
        self.root.destroy()

    # def run(self): # No longer needed, mainloop is called in __main__
    #     self.root.mainloop()

if __name__ == '__main__':
    import os
    import sys # For sys.platform
    from PIL import UnidentifiedImageError # Explicit import for exception handling
    root = tk.Tk()
    app = App(root)
    # Increased minimum height to better accommodate all control panel widgets
    root.minsize(900, 750)
    root.mainloop()
