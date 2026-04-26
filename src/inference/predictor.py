""" import pickle
import numpy as np

with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

def detect_gesture_from_landmarks(landmarks):
    try:
        print("Landmarks length:", len(landmarks))  # 👈 debug

        if len(landmarks) != 42:
            return "unknown"

        arr = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(arr)[0]
        return prediction

    except Exception as e:
        print("Error:", e)
        return "error" """

import pickle
import numpy as np
from collections import deque, Counter
import time


def normalize_landmarks(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y

    row = []
    for lm in landmarks:
        row.append(lm.x - base_x)
        row.append(lm.y - base_y)

    return row


class GesturePredictor:
    def __init__(self):
        try:
            self.model = pickle.load(open("models/gesture_model.pkl", "rb"))
            print("✅ ML model loaded")
        except Exception as e:
            self.model = None
            print("❌ Model load failed:", e)

        # 🔥 smoothing
        self.history = deque(maxlen=5)

        # 🔥 hold detection
        self.current_gesture = None
        self.last_change_time = time.time()
        self.hold_time = 1.0  # seconds

    def predict(self, landmarks):
        if self.model is None:
            return "no_model"

        try:
            # 🔥 normalize
            row = normalize_landmarks(landmarks)

            if len(row) != 42:
                return "invalid"

            arr = np.array(row).reshape(1, -1)

            # 🔥 get probabilities
            probs = self.model.predict_proba(arr)[0]
            pred = self.model.classes_[np.argmax(probs)]
            confidence = np.max(probs)

            # 🔥 confidence filter
            if confidence < 0.7:
                return self.current_gesture if self.current_gesture else "none"

            # 🔥 smoothing
            self.history.append(pred)
            stable_pred = Counter(self.history).most_common(1)[0][0]

            # 🔥 hold logic
            current_time = time.time()

            if stable_pred != self.current_gesture:
                if current_time - self.last_change_time > self.hold_time:
                    self.current_gesture = stable_pred
                    self.last_change_time = current_time

            return self.current_gesture if self.current_gesture else "none"

        except Exception as e:
            print("Prediction error:", e)
            return "error"