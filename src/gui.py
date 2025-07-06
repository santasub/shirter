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
        self.is_processing = False

        # HSV Color range defaults (e.g., for a shade of green)
        # User will be able to adjust this.
        self.lower_hsv = np.array([35, 100, 100]) # Increased min S and V
        self.upper_hsv = np.array([85, 255, 255])
        self.hsv_color_display = None # For showing selected color swatch

        # Main layout frames
        self.video_frame = ttk.Frame(self.root, padding="10")
        self.video_frame.grid(row=0, column=0, sticky="nsew")

        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.grid(row=0, column=1, sticky="nsew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        # --- Video Display Area ---
        # Set a default size for video labels to prevent collapsing
        self.default_video_bg = Image.new('RGB', (640, 480), (100, 100, 100))
        self.default_video_img = ImageTk.PhotoImage(self.default_video_bg)

        self.raw_video_label = ttk.Label(self.video_frame, image=self.default_video_img)
        self.raw_video_label.pack(pady=5, padx=5, expand=True, fill="both")

        self.processed_video_label = ttk.Label(self.video_frame, image=self.default_video_img)
        self.processed_video_label.pack(pady=5, padx=5, expand=True, fill="both")

        # --- Control Panel Area ---
        # Camera Selection
        camera_select_frame = ttk.LabelFrame(self.control_frame, text="Camera Setup")
        camera_select_frame.pack(fill="x", expand=False, padx=5, pady=5) # expand=False

        self.camera_var = tk.StringVar()
        self.available_cameras = self._list_available_cameras() # Populate here, before GUI elements use it

        camera_names = list(self.available_cameras.keys())
        initial_cam_name = ""
        cam_dropdown_state = "disabled"

        if camera_names:
            initial_cam_name = camera_names[0]
            cam_dropdown_state = "readonly"
        else:
            # If no cameras, add a dummy entry for display and keep disabled
            camera_names = ["No cameras found"]
            initial_cam_name = camera_names[0]
            # self.available_cameras will remain empty or be {'No cameras found': -1} from _list_available_cameras
            # Ensure self.available_cameras has this key if we rely on it later for index
            if not self.available_cameras: # If _list_available_cameras returned truly empty
                 self.available_cameras[initial_cam_name] = -1


        self.camera_var.set(initial_cam_name)
        self.camera_dropdown = ttk.OptionMenu(camera_select_frame, self.camera_var,
                                              initial_cam_name, *camera_names) # Pass initial_cam_name as the default
        self.camera_dropdown.pack(pady=5, padx=5, fill="x")
        self.camera_dropdown.config(state=cam_dropdown_state)

        # Color Selection (HSV)
        self.color_frame = ttk.LabelFrame(self.control_frame, text="Shirt Color (HSV)")
        self.color_frame.pack(fill="x", expand=False, padx=5, pady=5) # expand=False

        self.pick_color_button = ttk.Button(self.color_frame, text="Pick Shirt Color (from Preview)", command=self.enable_color_picker_mode)
        self.pick_color_button.pack(pady=2)
        self.color_picker_mode = False

        # Display for the chosen color
        self.hsv_color_display_label = ttk.Label(self.color_frame, text="Selected Color:")
        self.hsv_color_display_label.pack(pady=2)
        self.hsv_color_swatch = tk.Label(self.color_frame, background="grey", width=10, height=2)
        self.hsv_color_swatch.pack(pady=2)
        self._update_color_swatch(self.lower_hsv, self.upper_hsv) # Initial swatch

        # Manual HSV Sliders
        self.hsv_sliders = {}
        for i, label in enumerate(["H_low", "S_low", "V_low", "H_high", "S_high", "V_high"]):
            val = self.lower_hsv[i % 3] if "low" in label else self.upper_hsv[i % 3]
            max_val = 179 if "H_" in label else 255
            scale = tk.Scale(self.color_frame, from_=0, to=max_val, orient=tk.HORIZONTAL, label=label, length=200, command=self._update_hsv_from_sliders)
            scale.set(val)
            scale.pack(fill="x", padx=2)
            self.hsv_sliders[label] = scale

        # Text Overlay
        text_overlay_frame = ttk.LabelFrame(self.control_frame, text="Text Overlay")
        text_overlay_frame.pack(fill="x", expand=False, padx=5, pady=5) # expand=False

        self.text_entry_var = tk.StringVar(value="DJ SHIRT")
        self.text_entry = ttk.Entry(text_overlay_frame, textvariable=self.text_entry_var)
        self.text_entry.pack(pady=5, padx=5, fill="x")

        # Logo Overlay
        logo_overlay_frame = ttk.LabelFrame(self.control_frame, text="Logo Overlay")
        logo_overlay_frame.pack(fill="x", expand=False, padx=5, pady=5) # expand=False

        ttk.Button(logo_overlay_frame, text="Upload Logo", command=self.upload_logo).pack(pady=2)
        self.logo_path_var = tk.StringVar(value="No logo selected.")
        ttk.Label(logo_overlay_frame, textvariable=self.logo_path_var, wraplength=180).pack(pady=2, padx=5, fill="x")

        # ROI Definition (Placeholder)
        roi_frame = ttk.LabelFrame(self.control_frame, text="Overlay Regions (ROI)")
        roi_frame.pack(fill="x", expand=False, padx=5, pady=5) # expand=False
        ttk.Button(roi_frame, text="Define Shirt Regions", command=self.define_roi, state=tk.DISABLED).pack(pady=5) # Disabled for now

        # Start/Stop Button
        self.start_stop_button = ttk.Button(self.control_frame, text="Start Processing", command=self.toggle_processing)
        self.start_stop_button.pack(pady=10, padx=5, fill="x", side=tk.BOTTOM)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close

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

    def _update_hsv_from_sliders(self, event=None):
        self.lower_hsv[0] = self.hsv_sliders["H_low"].get()
        self.lower_hsv[1] = self.hsv_sliders["S_low"].get()
        self.lower_hsv[2] = self.hsv_sliders["V_low"].get()
        self.upper_hsv[0] = self.hsv_sliders["H_high"].get()
        self.upper_hsv[1] = self.hsv_sliders["S_high"].get()
        self.upper_hsv[2] = self.hsv_sliders["V_high"].get()
        self._update_color_swatch(self.lower_hsv, self.upper_hsv)

    def _update_color_swatch(self, lower_hsv, upper_hsv):
        # Create an average color for the swatch
        avg_h = int((lower_hsv[0] + upper_hsv[0]) / 2)
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
                hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
                print(f"Clicked BGR: {bgr_color}, HSV: {hsv_color}")

                # Set a small range around the picked H value, and broader S, V
                # These tolerances might need to be configurable
                hue_tolerance = 10
                saturation_tolerance_low = 50
                value_tolerance_low = 50
                saturation_tolerance_high = 255
                value_tolerance_high = 255


                self.lower_hsv[0] = max(0, hsv_color[0] - hue_tolerance)
                self.upper_hsv[0] = min(179, hsv_color[0] + hue_tolerance)
                self.lower_hsv[1] = max(0, hsv_color[1] - saturation_tolerance_low) # Or a fixed low like 50-100
                self.upper_hsv[1] = min(255, hsv_color[1] + saturation_tolerance_high) # Or fixed high 255
                self.lower_hsv[2] = max(0, hsv_color[2] - value_tolerance_low) # Or a fixed low like 50-100
                self.upper_hsv[2] = min(255, hsv_color[2] + value_tolerance_high) # Or fixed high 255

                # Update sliders
                self.hsv_sliders["H_low"].set(self.lower_hsv[0])
                self.hsv_sliders["S_low"].set(self.lower_hsv[1])
                self.hsv_sliders["V_low"].set(self.lower_hsv[2])
                self.hsv_sliders["H_high"].set(self.upper_hsv[0])
                self.hsv_sliders["S_high"].set(self.upper_hsv[1])
                self.hsv_sliders["V_high"].set(self.upper_hsv[2])

                self._update_color_swatch(self.lower_hsv, self.upper_hsv)

            self.color_picker_mode = False
            self.pick_color_button.config(text="Pick Shirt Color (from Preview)", state=tk.NORMAL)
            self.raw_video_label.unbind("<Button-1>")


    def upload_logo(self):
        filepath = filedialog.askopenfilename(
            title="Select Logo Image",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        if filepath:
            self.logo_path_var.set(filepath)
            print(f"Selected logo: {filepath}")
            # TODO: Load and store the image

    def define_roi(self):
        # Placeholder for ROI definition logic
        print("ROI definition button clicked. To be implemented.")
        # This will likely involve capturing a frame and drawing on it.

    def toggle_processing(self):
        self.is_processing = not self.is_processing
        if self.is_processing:
            cam_idx_str = self.camera_var.get()
            if not cam_idx_str or "No cameras found" in cam_idx_str :
                messagebox.showerror("Error", "No camera selected or available.")
                self.is_processing = False
                return

            cam_idx = self.available_cameras[cam_idx_str]
            if cam_idx == -1:
                messagebox.showerror("Error", "Invalid camera selected. Please ensure 'No cameras found' is not selected if cameras are available.")
                self.is_processing = False
                return

            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(cam_idx, backend)

            if not self.cap.isOpened():
                error_message = f"Could not open camera {cam_idx}."
                if sys.platform == 'darwin':
                    error_message += "\nPlease ensure the application has permission to access the camera. Check System Settings > Privacy & Security > Camera."
                messagebox.showerror("Error", error_message)
                self.is_processing = False
                self.cap = None
                return

            self.start_stop_button.config(text="Stop Processing")
            self.camera_dropdown.config(state="disabled") # Disable camera selection while processing
            self.pick_color_button.config(state="normal") # Enable color picking
            print("Processing Started")
            self._video_loop()
        else:
            self.start_stop_button.config(text="Start Processing")
            self.camera_dropdown.config(state="readonly" if self.available_cameras and next(iter(self.available_cameras)) != "No cameras found" else "disabled")
            self.pick_color_button.config(state="disabled")
            if self.color_picker_mode: # If color picking was active, disable it
                self.color_picker_mode = False
                self.raw_video_label.unbind("<Button-1>")

            print("Processing Stopped")
            if self.cap:
                self.cap.release()
                self.cap = None
            # Reset video labels to default image when stopping
            self.raw_video_label.configure(image=self.default_video_img)
            self.raw_video_label.image = self.default_video_img # Keep a reference
            self.processed_video_label.configure(image=self.default_video_img)
            self.processed_video_label.image = self.default_video_img # Keep a reference


    def _video_loop(self):
        if not self.is_processing or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?).")
            self.root.after(100, self._video_loop) # Try again shortly
            return

        # Store a copy for the color picker, before any overlays are drawn on `frame`
        self.current_raw_frame_for_picker = frame.copy()

        # --- Start Processing Logic (from main.py) ---
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Use current HSV range from sliders/picker
        current_lower_hsv = self.lower_hsv
        current_upper_hsv = self.upper_hsv

        mask = cv2.inRange(hsv_frame, current_lower_hsv, current_upper_hsv)

        # Optional: Morphological operations (can be added as a GUI option later)
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        processed_frame_display = frame.copy() # Frame to draw bounding boxes, text etc. on

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            min_contour_area = 500 # TODO: Make this configurable in GUI
            if cv2.contourArea(largest_contour) > min_contour_area:
                # --- Start Perspective Warp ---
                # 1. Get an approximate quadrilateral for the contour
                peri = cv2.arcLength(largest_contour, True)
                approx_quad = cv2.approxPolyDP(largest_contour, 0.02 * peri, True) # Adjust epsilon as needed

                # Draw the approximated contour
                # cv2.drawContours(processed_frame_display, [approx_quad], -1, (255, 0, 0), 3)


                # We need 4 points for perspective transform.
                # If approxPolyDP gives us 4 points, great. Otherwise, we might need more robust methods
                # or fall back to bounding box.
                if len(approx_quad) == 4:
                    target_points = approx_quad.reshape(4, 2).astype(np.float32)

                    # Order points: top-left, top-right, bottom-right, bottom-left
                    # This is crucial for cv2.getPerspectiveTransform
                    rect = np.zeros((4, 2), dtype="float32")
                    s = target_points.sum(axis=1)
                    rect[0] = target_points[np.argmin(s)]
                    rect[2] = target_points[np.argmax(s)]
                    diff = np.diff(target_points, axis=1)
                    rect[1] = target_points[np.argmin(diff)]
                    rect[3] = target_points[np.argmax(diff)]
                    ordered_target_points = rect

                    # 2. Create the text image to warp
                    text_to_overlay = self.text_entry_var.get()
                    if not text_to_overlay.strip(): # Don't process if text is empty
                        text_to_overlay = " " # Avoid errors with empty text size

                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5 # Made text larger for visibility
                    font_color_bgr = (0, 0, 0) # Black text
                    bg_color_bgr = (255, 255, 255) # White background for text texture
                    thickness = 2

                    # Estimate text size to create an appropriately sized source image
                    # Add some padding
                    (text_w, text_h), baseline = cv2.getTextSize(text_to_overlay, font_face, font_scale, thickness)
                    padding = 20
                    src_w = text_w + 2 * padding
                    src_h = text_h + baseline + 2 * padding

                    # Create source image (text texture)
                    text_texture = np.full((src_h, src_w, 3), bg_color_bgr, dtype=np.uint8)
                    text_origin_x = padding
                    text_origin_y = text_h + padding # cv2.putText origin is bottom-left
                    cv2.putText(text_texture, text_to_overlay, (text_origin_x, text_origin_y), font_face, font_scale, font_color_bgr, thickness, cv2.LINE_AA)

                    # 3. Define source points for the text image (a simple rectangle)
                    source_points = np.array([
                        [0, 0],
                        [src_w - 1, 0],
                        [src_w - 1, src_h - 1],
                        [0, src_h - 1]
                    ], dtype=np.float32)

                    # 4. Get the perspective transformation matrix
                    matrix = cv2.getPerspectiveTransform(source_points, ordered_target_points)

                    # 5. Warp the text image
                    warped_text_texture = cv2.warpPerspective(text_texture, matrix, (processed_frame_display.shape[1], processed_frame_display.shape[0]))

                    # 6. Create a mask for blending (where text is not background)
                    # For this simple case, white background means we make it transparent
                    mask_for_blending = cv2.inRange(warped_text_texture, np.array([0,0,0]), np.array([250,250,250])) # Non-white parts
                    mask_for_blending = cv2.bitwise_not(mask_for_blending) # Invert, so text is white, bg is black for mask

                    # Blend
                    # Black out the area of text on the original frame
                    processed_frame_display = cv2.bitwise_and(processed_frame_display, processed_frame_display, mask=cv2.bitwise_not(mask_for_blending))
                    # Add the warped text
                    processed_frame_display = cv2.bitwise_or(processed_frame_display, warped_text_texture, mask=mask_for_blending)

                else: # Fallback to bounding box text if not 4 points
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(processed_frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box
                    text_to_overlay = self.text_entry_var.get()
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_color = (255, 255, 255) # White
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(text_to_overlay, font_face, font_scale, thickness)
                    text_x = x + (w - text_width) // 2
                    text_y = y + (h + text_height) // 2
                    cv2.putText(processed_frame_display, text_to_overlay, (text_x, text_y), font_face, font_scale, font_color, thickness, cv2.LINE_AA)
                # --- End Perspective Warp ---


        # --- End Processing Logic ---

        # Convert frames for Tkinter display
        # Raw frame (original from camera)
        img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_raw = Image.fromarray(img_raw)
        imgtk_raw = ImageTk.PhotoImage(image=img_raw)
        self.raw_video_label.imgtk = imgtk_raw
        self.raw_video_label.configure(image=imgtk_raw)

        # Processed frame (with mask or overlays)
        # For now, let's display the mask on the processed_video_label
        # Later, this will be frame_with_overlays
        img_processed = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) # Display mask
        # To display the frame with bounding box and warped text:
        img_processed = cv2.cvtColor(processed_frame_display, cv2.COLOR_BGR2RGB)

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
            self.cap.release()
        self.root.destroy()

    # def run(self): # No longer needed, mainloop is called in __main__
    #     self.root.mainloop()

if __name__ == '__main__':
    import os
    import sys # For sys.platform
    root = tk.Tk()
    app = App(root)
    root.minsize(900, 700)
    root.mainloop()
