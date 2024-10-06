import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize the webcam feed and set the dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width of the window
cap.set(4, 720)   # Height of the window

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# List to store drawing points [(x, y, color, brush_thickness)]
draw_points = []

# Eraser mode toggle
eraser_mode = False

# Scroll variables
scroll_mode = False
scroll_start_y = None

# Color options for drawing
paint_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
current_color = paint_colors[0]  # Initial color: Red
brush_thickness = 10
eraser_thickness = 50  # Eraser thickness

# UI Elements for Color Selection + Brush Size Controls
# Organized in a table-like grid
button_positions = {
    "color_rects": [(50, 50, 150, 150), (200, 50, 300, 150), (350, 50, 450, 150)],  # Color boxes
    "brush_increase_button": (500, 50, 600, 150),  # Increase brush size
    "brush_decrease_button": (650, 50, 750, 150),  # Decrease brush size
    "eraser_button": (800, 50, 950, 150),  # Eraser button
}

color_labels = ["Red", "Green", "Blue"]

# Variables for controlling the state of writing
drawing = False  # Flag to start drawing when the index finger is up

while True:
    # Capture frame from the webcam
    success, img = cap.read()

    # Flip the image horizontally for a mirror view
    img = cv2.flip(img, 1)

    # Copy the original frame as background (used for erasing)
    img_copy = img.copy()

    # Detect the hand and get the landmarks
    hands, img = detector.findHands(img)

    # Draw the color selection buttons
    for i, (x1, y1, x2, y2) in enumerate(button_positions["color_rects"]):
        color = paint_colors[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)  # Color boxes
        cv2.putText(img, color_labels[i], (x1 + 10, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Draw the brush size control buttons
    cv2.rectangle(img, button_positions["brush_increase_button"][:2], button_positions["brush_increase_button"][2:], (0, 255, 255), cv2.FILLED)  # Yellow box for increasing size
    cv2.putText(img, "+ Size", (button_positions["brush_increase_button"][0] + 10, button_positions["brush_increase_button"][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.rectangle(img, button_positions["brush_decrease_button"][:2], button_positions["brush_decrease_button"][2:], (0, 255, 255), cv2.FILLED)  # Yellow box for decreasing size
    cv2.putText(img, "- Size", (button_positions["brush_decrease_button"][0] + 10, button_positions["brush_decrease_button"][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # Eraser button
    cv2.rectangle(img, button_positions["eraser_button"][:2], button_positions["eraser_button"][2:], (255, 255, 255), cv2.FILLED)
    cv2.putText(img, "Eraser", (button_positions["eraser_button"][0] + 10, button_positions["eraser_button"][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # If a hand is detected
    if hands:
        # Get the first hand detected
        hand = hands[0]

        # Get the landmarks for the detected hand
        lmList = hand["lmList"]

        # Get the index finger tip (Landmark 8 is the index finger tip)
        index_finger_tip = lmList[8][:2]

        # Get the status of all fingers (1: finger up, 0: finger down)
        fingers = detector.fingersUp(hand)

        # Scroll Mode: If fist is detected (all fingers down)
        if fingers == [0, 0, 0, 0, 0]:
            if not scroll_mode:
                scroll_start_y = index_finger_tip[1]  # Start scrolling
                scroll_mode = True
        else:
            scroll_mode = False

        # Handle scrolling
        if scroll_mode and scroll_start_y is not None:
            # Calculate how much to scroll
            scroll_delta = index_finger_tip[1] - scroll_start_y

            # Shift all drawing points vertically
            for i in range(len(draw_points)):
                if draw_points[i] is not None:
                    point, color, size = draw_points[i]
                    new_point = (point[0], point[1] + int(scroll_delta))  # Shift the y-coordinate by scroll_delta
                    draw_points[i] = (new_point, color, size)

            # Reset scroll_start_y to prevent continuous shifting
            scroll_start_y = index_finger_tip[1]

        # Drawing Mode: If only the index finger is up
        if fingers[1] == 1 and fingers[2] == 0:
            if eraser_mode:
                # Erase the points under the eraser
                for i in range(len(draw_points) - 1, -1, -1):
                    if draw_points[i] is not None:
                        point, color, size = draw_points[i]
                        if point and np.linalg.norm(np.array(index_finger_tip) - np.array(point)) < eraser_thickness:
                            draw_points.pop(i)  # Remove points within the eraser radius
            else:
                draw_points.append((index_finger_tip, current_color, brush_thickness))
            drawing = True
        else:
            draw_points.append(None)  # Stop drawing if no finger is up
            drawing = False

        # Color Change: If index finger is inside color selection boxes
        for i, (x1, y1, x2, y2) in enumerate(button_positions["color_rects"]):
            if x1 < index_finger_tip[0] < x2 and y1 < index_finger_tip[1] < y2:
                current_color = paint_colors[i]
                eraser_mode = False  # Disable eraser when a color is selected

        # Brush Size Increase/Decrease Logic
        if button_positions["brush_increase_button"][0] < index_finger_tip[0] < button_positions["brush_increase_button"][2] and button_positions["brush_increase_button"][1] < index_finger_tip[1] < button_positions["brush_increase_button"][3]:
            brush_thickness += 2
        if button_positions["brush_decrease_button"][0] < index_finger_tip[0] < button_positions["brush_decrease_button"][2] and button_positions["brush_decrease_button"][1] < index_finger_tip[1] < button_positions["brush_decrease_button"][3]:
            brush_thickness = max(2, brush_thickness - 2)

        # Toggle Eraser Mode: If the index finger is inside the eraser button
        if button_positions["eraser_button"][0] < index_finger_tip[0] < button_positions["eraser_button"][2] and button_positions["eraser_button"][1] < index_finger_tip[1] < button_positions["eraser_button"][3]:
            eraser_mode = True

        # Clear Screen: Using thumbs-down gesture
        # When the thumb is down and all other fingers are up: [0, 1, 1, 1, 1]
        if fingers == [0, 1, 1, 1, 1]:
            draw_points.clear()  # Clear all drawing points

    # Drawing points on the screen
    for i in range(1, len(draw_points)):
        if draw_points[i] is not None and draw_points[i - 1] is not None:
            point1, color, size = draw_points[i - 1]
            point2, color, size = draw_points[i]
            cv2.line(img, point1, point2, color, size)

    # Show active color and brush size
    cv2.putText(img, "Active Color", (800, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.circle(img, (1050, 250), brush_thickness, current_color, cv2.FILLED)
    cv2.putText(img, f"Brush Size: {brush_thickness}", (1000, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the camera feed with drawing
    cv2.imshow("Virtual Writing with Scroll and Clear Screen", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
