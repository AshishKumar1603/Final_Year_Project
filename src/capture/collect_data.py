import cv2
import mediapipe as mp
import csv
import os
import time
import math

# 📁 ensure dataset folder
os.makedirs("datasets", exist_ok=True)

# 🔥 MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
gesture = input("Enter gesture name: ").strip().lower().replace(" ", "_")
file_path = "datasets/gestures.csv"
last_capture_time = 0
capture_delay = 0.4
with open(file_path, "a", newline="") as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                current_time = time.time()

                if current_time - last_capture_time > capture_delay:
                    row = []

                    #  NORMALIZATION
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    #  normalized coordinates
                    for lm in hand_landmarks.landmark:
                        row.append(lm.x - base_x)
                        row.append(lm.y - base_y)
                    #  distance features 
                    for lm in hand_landmarks.landmark:
                        dist = math.sqrt((lm.x - base_x)**2 + (lm.y - base_y)**2)
                        row.append(dist)

                    # label
                    row.append(gesture)

                    writer.writerow(row)

                    last_capture_time = current_time
                    print("Captured:", gesture)

                # 🎯 draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # 🖼 UI
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()