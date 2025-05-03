import cv2
import numpy as np
from norfair import Detection, Tracker, draw_tracked_objects
def rgba_to_hsv(rgba):
    bgr = np.uint8([[[rgba[2], rgba[1], rgba[0]]]])  # Convert RGBA to BGR (drop alpha)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

# Convert both RGBA colors to HSV
color1_hsv = rgba_to_hsv((117, 79, 67, 255))
color2_hsv = rgba_to_hsv((41, 28, 18, 255))

print("Color 1 HSV:", color1_hsv)
print("Color 2 HSV:", color2_hsv)

# Now create min/max HSV ranges
lower_hsv = np.minimum(color1_hsv, color2_hsv) - np.array([10, 40, 40])  # allow some flexibility
upper_hsv = np.maximum(color1_hsv, color2_hsv) + np.array([10, 40, 40])

# Clip to valid range
lower_hsv = np.clip(lower_hsv, 0, 255)
upper_hsv = np.clip(upper_hsv, 0, 255)

print("Lower HSV range:", lower_hsv)
print("Upper HSV range:", upper_hsv)

tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Video capture
cap = cv2.VideoCapture("basketball_game.mp4")  # or use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color filtering (tune values for your basketball)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            center = np.array([x + w // 2, y + h // 2])
            detections.append(Detection(points=center))

    tracked_objects = tracker.update(detections=detections)

    # Draw the tracking
    draw_tracked_objects(frame, tracked_objects)

    cv2.imshow("Basketball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()