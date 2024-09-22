import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create a blank canvas for drawing
canvas = None
drawing = False  # Variable to track if we are drawing
current_color = (255, 0, 0)  # Default color: Blue
brush_thickness = 7  # Brush thickness
shape = 'line'  # Default shape is line (instead of dots)
erase_mode = False  # Toggle erase mode
prev_point = None  # Track previous index finger point for continuous drawing

# Button parameters
button_height = 50
button_width = 150
buttons = {'Erase': (10, 10), 'Color': (170, 10), 'Shape': (330, 10)}

# Define colors for color change (red, green, blue, black)
color_options = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]
color_index = 0

# Define shapes (circle, rectangle, line)
shapes = ['circle', 'rectangle', 'line']
shape_index = 2  # Default to "line"

# Function to draw buttons
def draw_buttons(img):
    for name, (x, y) in buttons.items():
        cv2.rectangle(img, (x, y), (x + button_width, y + button_height), (255, 255, 255), -1)
        cv2.putText(img, name, (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Function to check if index finger is selecting a button
def check_button_press(cx, cy):
    global erase_mode, current_color, shape, drawing, color_index, shape_index
    for name, (x, y) in buttons.items():
        if x < cx < x + button_width and y < cy < y + button_height:
            if name == 'Erase':
                erase_mode = True
            elif name == 'Color':
                erase_mode = False
                color_index = (color_index + 1) % len(color_options)
                current_color = color_options[color_index]
            elif name == 'Shape':
                erase_mode = False
                shape_index = (shape_index + 1) % len(shapes)
                shape = shapes[shape_index]
            drawing = False  # Turn off drawing when a button is pressed

# Function to increase brightness
def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)  # Limit values to the range [0, 255]
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

# Capture Video from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror-like effect
    img = cv2.flip(img, 1)

    # Increase brightness for better visibility
    img = increase_brightness(img, value=50)

    # Convert the image color to RGB (Mediapipe expects RGB images)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(img_rgb)

    # Create the canvas when the first frame is captured
    if canvas is None:
        canvas = np.zeros_like(img)

    # Draw buttons on the screen
    draw_buttons(img)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Get the pixel coordinates of the index finger tip
            h, w, _ = img.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw the hand landmarks on the video feed
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if we are hovering over a button
            check_button_press(cx, cy)

            # Check if the index finger tip is moving (simulating "writing")
            if drawing and not erase_mode:
                if shape == 'circle':
                    cv2.circle(canvas, (cx, cy), brush_thickness, current_color, -1)  # Draw a circle
                elif shape == 'rectangle':
                    cv2.rectangle(canvas, (cx - 20, cy - 20), (cx + 20, cy + 20), current_color, -1)  # Draw a rectangle
                elif shape == 'line' and prev_point is not None:
                    # Draw a line from the previous point to the current point
                    cv2.line(canvas, prev_point, (cx, cy), current_color, brush_thickness)

            # Erase mode
            if erase_mode:
                cv2.circle(canvas, (cx, cy), 50, (0, 0, 0), -1)  # Erase by drawing black circles

            # Track previous point for continuous drawing
            prev_point = (cx, cy) if not erase_mode else None

            # Toggle drawing mode by thumb position
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_cx, thumb_cy = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # If thumb and index are close together, stop drawing
            if np.hypot(cx - thumb_cx, cy - thumb_cy) < 40:
                drawing = False
                prev_point = None  # Reset the previous point when not drawing
            else:
                drawing = True

    # Merge the canvas with the webcam feed
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    # Display the output
    cv2.imshow("Air Writing with Operations", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()   
cv2.destroyAllWindows()
