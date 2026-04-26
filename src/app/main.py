from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
import time
import math

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load ML model
with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ MediaPipe init (improved)
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 🔥 smoothing + hold
history = deque(maxlen=5)
current_gesture = "none"
last_change_time = time.time()
hold_time = 0.7   # faster response


# ✅ Home
@app.get("/")
def home():
    return {"message": "Gesture API running"}


# 🔥 MAIN: IMAGE PREDICTION API
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    global current_gesture, last_change_time

    try:
        contents = await file.read()

        # bytes → image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark

            # 🔥 normalization
            base_x = landmarks[0].x
            base_y = landmarks[0].y

            row = []

            # ✅ (1) normalized coordinates
            for lm in landmarks:
                row.append(lm.x - base_x)
                row.append(lm.y - base_y)

            # 🔥 (2) distance features (CRITICAL FIX)
            for lm in landmarks:
                dist = math.sqrt((lm.x - base_x)**2 + (lm.y - base_y)**2)
                row.append(dist)

            # ✅ now should be 63 features
            if len(row) != 63:
                return {"gesture": current_gesture}

            arr = np.array(row).reshape(1, -1)

            # 🔥 prediction + confidence
            probs = model.predict_proba(arr)[0]
            pred = model.classes_[np.argmax(probs)]
            confidence = np.max(probs)

            print("PRED:", pred, "CONF:", confidence)

            # 🔥 confidence filter (tuned)
            if confidence < 0.6:
                return {"gesture": current_gesture}

            # 🔥 smoothing
            history.append(pred)
            stable_pred = Counter(history).most_common(1)[0][0]

            # 🔥 hold detection
            current_time = time.time()
            if stable_pred != current_gesture:
                if current_time - last_change_time > hold_time:
                    current_gesture = stable_pred
                    last_change_time = current_time

            return {"gesture": current_gesture}

        return {"gesture": "none"}

    except Exception as e:
        print("Error:", e)
        return {"gesture": "error"}