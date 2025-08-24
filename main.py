import cv2
import numpy as np


def print_controls():
    """Prints the control keys to the console."""
    print("--- Controls ---")
    print(" q: Quit")
    print(" p: Toggle Projection/Debug mode")
    print(" h: Show this help message")
    print(" i, k, j, l: Move overlay (up, down, left, right)")
    print(" u, o: Increase/Decrease move step")
    print(" r: Reset overlay position")
    print("----------------")


def main():
    """
    Main function to run the face tracking and overlay application.
    """
    # Use the cv2.data.haarcascades path for robustness
    cascade_filename = 'haarcascade_frontalface_default.xml'
    face_cascade_path = cv2.data.haarcascades + cascade_filename
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {face_cascade_path}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # --- Control settings ---
    projection_mode = False
    offset_x, offset_y = 0, 0
    move_step = 5
    print_controls()

    # --- Overlay settings ---
    overlay_text = "Tracking..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White
    thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Create the output canvas based on the current mode
        if projection_mode:
            output_canvas = np.zeros_like(frame)
        else:
            output_canvas = frame.copy()

        # Face detection runs on the original frame's grayscale version
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate base text position
            text_size = cv2.getTextSize(
                overlay_text, font, font_scale, thickness)[0]
            base_text_x = center_x - text_size[0] // 2
            base_text_y = y - 10
            if base_text_y < text_size[1]:
                base_text_y = y + h + text_size[1] + 10

            # Apply offset
            final_text_x = base_text_x + offset_x
            final_text_y = base_text_y + offset_y

            cv2.putText(output_canvas, overlay_text,
                        (final_text_x, final_text_y),
                        font, font_scale, font_color, thickness)

            # Draw debug visuals only in debug mode
            if not projection_mode:
                cv2.rectangle(output_canvas, (x, y), (x+w, y+h),
                              (255, 0, 0), 2)
                cv2.circle(output_canvas, (center_x, center_y),
                           5, (0, 255, 0), -1)

        cv2.imshow('Output', output_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            projection_mode = not projection_mode
            mode = "Projection" if projection_mode else "Debug"
            print(f"Switched to {mode} Mode")
        elif key == ord('h'):
            print_controls()
        elif key == ord('j'):  # left
            offset_x -= move_step
        elif key == ord('l'):  # right
            offset_x += move_step
        elif key == ord('i'):  # up
            offset_y -= move_step
        elif key == ord('k'):  # down
            offset_y += move_step
        elif key == ord('r'):  # reset
            offset_x, offset_y = 0, 0
            print("Overlay position reset.")
        elif key == ord('u'):  # step up
            move_step += 1
            print(f"Move step is now: {move_step}")
        elif key == ord('o'):  # step down
            move_step = max(1, move_step - 1)
            print(f"Move step is now: {move_step}")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
