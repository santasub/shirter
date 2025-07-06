import cv2
import numpy as np

def list_available_cameras():
    """Lists available camera devices and their indices."""
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # Use CAP_DSHOW for Windows, can be different for other OS
        if not cap.isOpened():
            cap.release()
            break
        arr.append(index)
        cap.release()
        index += 1
    if not arr:
        print("No cameras found!")
    else:
        print("Available camera indices:", arr)
    return arr

def main(camera_index=0):
    """
    Captures video from the specified camera index and displays it.
    Press 'q' to quit.
    """
    print(f"Attempting to use camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open video device with index {camera_index}.")
        print("If you have other cameras, try changing the camera_index.")
        print("Available cameras will be listed if any are found.")
        list_available_cameras()
        return

    print("Video capture started. Press 'q' to quit.")

    # Placeholder HSV color range for a common color (e.g., a shade of green)
    # These will eventually be user-configurable
    # H: 0-179, S: 0-255, V: 0-255 in OpenCV
    lower_hsv = (35, 50, 50)  # Lower bound for green
    upper_hsv = (85, 255, 255) # Upper bound for green

    cv2.namedWindow('Live Video Feed')
    cv2.namedWindow('Color Mask')

    # Simple trackbars for HSV adjustment (will be replaced by GUI elements later)
    def nothing(x):
        pass

    cv2.createTrackbar('H_low', 'Color Mask', lower_hsv[0], 179, nothing)
    cv2.createTrackbar('S_low', 'Color Mask', lower_hsv[1], 255, nothing)
    cv2.createTrackbar('V_low', 'Color Mask', lower_hsv[2], 255, nothing)
    cv2.createTrackbar('H_high', 'Color Mask', upper_hsv[0], 179, nothing)
    cv2.createTrackbar('S_high', 'Color Mask', upper_hsv[1], 255, nothing)
    cv2.createTrackbar('V_high', 'Color Mask', upper_hsv[2], 255, nothing)

    # --- PyVirtualCam Placeholder Initialization ---
    # import pyvirtualcam # Add to imports at the top
    # virtual_cam = None
    # try:
    #     # Get frame dimensions from camera
    #     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     cam_fps = cap.get(cv2.CAP_PROP_FPS)
    #     if cam_fps == 0: # Some cameras might not report FPS correctly
    #         cam_fps = 30 # Default to 30 if not available
    #
    #     # Ensure width and height are not zero
    #     if frame_width > 0 and frame_height > 0:
    #         # virtual_cam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=cam_fps, fmt=pyvirtualcam.PixelFormat.BGR)
    #         # print(f"Virtual camera initialized: {virtual_cam.device}")
    #         print(f"Frames will be sent to virtual camera with resolution {frame_width}x{frame_height} @ {cam_fps} FPS (Conceptual)")
    #     else:
    #         print("Warning: Frame width or height is 0. Cannot initialize virtual camera.")
    #
    # except Exception as e:
    #     print(f"Error initializing virtual camera (conceptual): {e}")
    #     print("Ensure pyvirtualcam is installed and a backend (OBS Virtual Cam, CamTwist, v4l2loopback) is available.")
    # --- End PyVirtualCam Placeholder ---


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Make a copy of the frame if you intend to send the original to virtual cam before drawing UI elements for it
        # processed_frame = frame.copy() # If overlays are drawn on 'frame', then 'frame' is the processed_frame

        # Get current positions of trackbars
        h_low = cv2.getTrackbarPos('H_low', 'Color Mask')
        s_low = cv2.getTrackbarPos('S_low', 'Color Mask')
        v_low = cv2.getTrackbarPos('V_low', 'Color Mask')
        h_high = cv2.getTrackbarPos('H_high', 'Color Mask')
        s_high = cv2.getTrackbarPos('S_high', 'Color Mask')
        v_high = cv2.getTrackbarPos('V_high', 'Color Mask')

        current_lower_hsv = (h_low, s_low, v_low)
        current_upper_hsv = (h_high, s_high, v_high)

        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask based on the HSV range
        mask = cv2.inRange(hsv_frame, current_lower_hsv, current_upper_hsv)

        # Optional: Apply some morphological operations to reduce noise
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=1)
        # mask = cv2.dilate(mask, kernel, iterations=1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the shirt
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Filter by area to avoid tiny noise spots being chosen
            min_contour_area = 500 # This value might need adjustment
            if cv2.contourArea(largest_contour) > min_contour_area:
                # Get the bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box

                # --- Add Text Overlay ---
                text_to_overlay = "DJ SHIRT"
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_color = (255, 255, 255) # White
                thickness = 2
                line_type = cv2.LINE_AA

                # Get text size to help center it
                (text_width, text_height), baseline = cv2.getTextSize(text_to_overlay, font_face, font_scale, thickness)

                # Calculate text position (center of the bounding box)
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2 # Adjusted for baseline to be more centered vertically

                # Put text on the frame
                cv2.putText(frame, text_to_overlay, (text_x, text_y), font_face, font_scale, font_color, thickness, line_type)
                # --- End Text Overlay ---

                # Optionally, draw the contour itself
                # cv2.drawContours(frame, [largest_contour], -1, (0,0,255), 2)


        cv2.imshow('Live Video Feed', frame) # Now with potential bounding box and text
        cv2.imshow('Color Mask', mask)

        # --- PyVirtualCam Placeholder Frame Sending ---
        # if virtual_cam:
        #     try:
        #         # Assume 'frame' is the final processed frame to be sent
        #         # Ensure frame is BGR if that's what virtual_cam is expecting
        #         virtual_cam.send(frame)
        #         virtual_cam.sleep_until_next_frame() # To maintain FPS
        #     except Exception as e:
        #         print(f"Error sending frame to virtual camera: {e}")
        #         # Potentially stop trying to send frames or re-initialize
        #         # virtual_cam = None # Simple way to stop trying
        # --- End PyVirtualCam Placeholder ---


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video capture stopped and windows closed.")

    # --- PyVirtualCam Placeholder Cleanup ---
    # if virtual_cam:
    #     print("Closing virtual camera.")
    #     # virtual_cam.close() # Or use 'with' statement for automatic closing
    # --- End PyVirtualCam Placeholder ---


if __name__ == '__main__':
    # We'll need numpy for morphological operations if we add them
    # import numpy as np # Moved to top if used globally or keep here for main-only
    available_cameras = list_available_cameras()
    selected_camera_index = 0 # Default to 0
    if not available_cameras:
        print("Exiting as no cameras are available.")
    else:
        # Simple way to select camera if multiple are available
        # In a GUI, this would be a dropdown or selection list
        if len(available_cameras) > 1:
            try:
                user_choice = input(f"Enter camera index to use (e.g., {available_cameras[0]}): ")
                selected_camera_index = int(user_choice)
                if selected_camera_index not in available_cameras:
                    print(f"Invalid index. Defaulting to {available_cameras[0]}.")
                    selected_camera_index = available_cameras[0]
            except ValueError:
                print(f"Invalid input. Defaulting to {available_cameras[0]}.")
                selected_camera_index = available_cameras[0]

        main(camera_index=selected_camera_index)
    print("Application finished.")
