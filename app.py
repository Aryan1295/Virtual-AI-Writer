from flask import Flask
import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
import threading
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize variables
mouse = Controller()
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

drawing_mode = False  # Toggle for drawing mode
prev_x, prev_y = None, None  # Previous coordinates for drawing
drawing_canvas = np.zeros((480, 640, 3), np.uint8)  # Canvas for drawing

# Erase button properties
button_width, button_height = 150, 50
button_x, button_y = 20, 20  # Top-left corner
button_color = (0, 0, 255)  # Red color for the button
button_thickness = 2

# Ink color button properties
color_button_width, color_button_height = 150, 50
color_button_x, color_button_y = button_x + button_width + 20, button_y  # Positioned beside the erase button
color_button_color = (0, 255, 0)  # Green color for the button
color_button_thickness = 2

# Shape button properties
shape_button_width, shape_button_height = 150, 50
circle_button_x, circle_button_y = color_button_x + color_button_width + 20, button_y  # Positioned beside the color button
triangle_button_x, triangle_button_y = circle_button_x + shape_button_width + 20, button_y
rectangle_button_x, rectangle_button_y = triangle_button_x + shape_button_width + 20, button_y

def get_darker_color():
    # Generate a random color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Reduce the brightness to make it darker
    return tuple(int(c * 0.5) for c in color)  # Adjust the scaling factor as needed

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 50
    )

def is_double_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50
    )

def is_screenshot(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )

def is_thumbs_up(landmark_list):
    return (
        landmark_list[4][1] < landmark_list[3][1] < landmark_list[2][1] and
        all(landmark_list[i][1] > landmark_list[i-2][1] for i in range(8, 21, 4))
    )

def is_thumbs_down(landmark_list):
    return (
        landmark_list[4][1] > landmark_list[3][1] > landmark_list[2][1] and
        all(landmark_list[i][1] > landmark_list[i-2][1] for i in range(8, 21, 4))
    )

def is_pinch(landmark_list, thumb_index_dist):
    # This function checks if the thumb and index finger are close to each other, indicating a pinch gesture
    return thumb_index_dist < 50  # Adjust the distance threshold as needed

pinch_active = False
current_color=(255,255,255)

def detect_gesture(frame, landmark_list, processed):
    global drawing_mode, prev_x, prev_y, drawing_canvas, current_color, pinch_active

    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[8]])

        if is_thumbs_up(landmark_list):
            drawing_mode = True
            pinch_active = False  # Ensure pinch is not active when drawing mode is enabled
            cv2.putText(frame, "Drawing Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif is_thumbs_down(landmark_list):
            drawing_mode = False
            prev_x, prev_y = None, None  # Reset previous drawing position
            pinch_active = False  # Ensure pinch is not active when drawing mode is disabled
            cv2.putText(frame, "Virtual Mouse Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw erase button
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, button_thickness)
        cv2.putText(frame, "Erase", (button_x + 20, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw color changing button
        cv2.rectangle(frame, (color_button_x, color_button_y), (color_button_x + color_button_width, color_button_y + color_button_height), color_button_color, color_button_thickness)
        cv2.putText(frame, "Color", (color_button_x + 20, color_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if drawing_mode:
            if is_pinch(landmark_list, thumb_index_dist):
                pinch_active = True  # Set pinch as active
                cv2.putText(frame, "Drawing Stopped (Pinch Gesture)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                if pinch_active:
                    pinch_active = False  # Deactivate pinch gesture when it's released
                if index_finger_tip and not pinch_active:
                    x = int(index_finger_tip.x * frame.shape[1])
                    y = int(index_finger_tip.y * frame.shape[0])
                    if prev_x is not None and prev_y is not None:
                        cv2.line(drawing_canvas, (prev_x, prev_y), (x, y), current_color, 5)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None  # Reset if no finger detected
        else:
            if util.get_distance([landmark_list[4], landmark_list[5]]) < 50 and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
                move_mouse(index_finger_tip)
            elif is_left_click(landmark_list, thumb_index_dist):
                mouse.press(Button.left)
                mouse.release(Button.left)
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_right_click(landmark_list, thumb_index_dist):
                mouse.press(Button.right)
                mouse.release(Button.right)
                cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_double_click(landmark_list, thumb_index_dist):
                pyautogui.doubleClick()
                cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif is_screenshot(landmark_list, thumb_index_dist):
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f'my_screenshot_{label}.png')
                cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Check if the finger is within the erase button area
        if index_finger_tip:
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])
            if (button_x < x < button_x + button_width) and (button_y < y < button_y + button_height):
                drawing_canvas[:] = 0  # Clear the drawing canvas
                cv2.putText(frame, "Canvas Cleared", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif (color_button_x < x < color_button_x + color_button_width) and (color_button_y < y < color_button_y + color_button_height):
                # Toggle ink color
                current_color = get_darker_color()
                cv2.putText(frame, "Color Changed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw shape buttons
        cv2.rectangle(frame, (circle_button_x, circle_button_y), (circle_button_x + shape_button_width, circle_button_y + shape_button_height), (255, 0, 0), 2)
        cv2.putText(frame, "Circle", (circle_button_x + 20, circle_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (triangle_button_x, triangle_button_y), (triangle_button_x + shape_button_width, triangle_button_y + shape_button_height), (0, 255, 0), 2)
        cv2.putText(frame, "Triangle", (triangle_button_x + 20, triangle_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (rectangle_button_x, rectangle_button_y), (rectangle_button_x + shape_button_width, rectangle_button_y + shape_button_height), (0, 0, 255), 2)
        cv2.putText(frame, "Rectangle", (rectangle_button_x + 20, rectangle_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if index_finger_tip:
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])
            if (circle_button_x < x < circle_button_x + shape_button_width) and (circle_button_y < y < circle_button_y + shape_button_height):
                # Draw circle as sticker
                cv2.circle(drawing_canvas, (x, y), 50, current_color, -1)  # Draw a filled circle
                cv2.putText(frame, "Drawing Circle", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif (triangle_button_x < x < triangle_button_x + shape_button_width) and (triangle_button_y < y < triangle_button_y + shape_button_height):
                # Draw triangle as sticker
                pts = np.array([[x, y-50], [x-50, y+50], [x+50, y+50]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(drawing_canvas, [pts], isClosed=True, color=current_color, thickness=5)
                cv2.putText(frame, "Drawing Triangle", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif (rectangle_button_x < x < rectangle_button_x + shape_button_width) and (rectangle_button_y < y < rectangle_button_y + shape_button_height):
                # Draw rectangle as sticker
                cv2.rectangle(drawing_canvas, (x-50, y-25), (x+50, y+25), current_color, -1)  # Draw a filled rectangle
                cv2.putText(frame, "Drawing Rectangle", (50, 100), cv2.FONT_HERSHEY_SIMPLE, 1, (255, 255, 255), 2)


def camera_loop():
    cap = cv2.VideoCapture(0)

    global drawing_canvas

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frame_rgb)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            # Overlay drawing canvas on the frame
            frame_with_drawing = cv2.addWeighted(frame, 1, drawing_canvas, 0.5, 0)
            cv2.imshow('Frame', frame_with_drawing)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return "Virtual Mouse and Air Drawing Running in Background"

if __name__ == '__main__':
    # Run the camera loop in a separate thread
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    # Run the Flask server
    app.run(host='0.0.0.0', port=5000)
