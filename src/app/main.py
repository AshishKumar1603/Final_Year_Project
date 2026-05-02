from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter, defaultdict
import time
import math
from ultralytics import YOLO

app = FastAPI()

# =========================

# CORS

# =========================

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# =========================

# LOAD MODELS

# =========================

with open("models/gesture_model.pkl", "rb") as f:
   gesture_model = pickle.load(f)

yolo_model = YOLO("yolov8n.pt")

# =========================

# MEDIAPIPE

# =========================

mp_hands = mp.solutions.hands.Hands(
static_image_mode=False,
max_num_hands=1,
min_detection_confidence=0.6,
min_tracking_confidence=0.6
)

# =========================

# STATE

# =========================

history = deque(maxlen=5)
current_gesture = "none"
last_change_time = time.time()
hold_time = 0.7

track_history = defaultdict(lambda: deque(maxlen=5))

# =========================

# 🔥 UPDATED YOLO FUNCTION

# =========================

def detect_objects(frame):
   results = yolo_model.track(frame, persist=True, conf=0.4)


   detected_objects = []

   if results[0].boxes is not None:

      boxes = results[0].boxes.xyxy
      ids = results[0].boxes.id
      classes = results[0].boxes.cls

      if ids is not None:
          ids = ids.int().cpu().tolist()
      else:
          ids = [0] * len(boxes)

      for box, track_id, cls in zip(boxes, ids, classes):
          box = box.cpu().numpy()

          # 🔥 smoothing
          history_box = track_history[track_id]
          history_box.append(box)
          avg_box = sum(history_box) / len(history_box)

          x1, y1, x2, y2 = avg_box

          label = yolo_model.names[int(cls)]

          detected_objects.append({
              "label": label,
              "box": [float(x1), float(y1), float(x2), float(y2)]
          })

   return detected_objects


# =========================

# ROUTES

# =========================

@app.get("/")
def home():
    return {"message": "Gesture + YOLO API running"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
   global current_gesture, last_change_time


   try:
       contents = await file.read()

       nparr = np.frombuffer(contents, np.uint8)
       frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

       # 🔥 YOLO
       objects = detect_objects(frame)

       # 🔥 GESTURE
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       result = mp_hands.process(rgb)

       if result.multi_hand_landmarks:
           landmarks = result.multi_hand_landmarks[0].landmark

           base_x = landmarks[0].x
           base_y = landmarks[0].y

           row = []

           # normalized coords
           for lm in landmarks:
               row.append(lm.x - base_x)
               row.append(lm.y - base_y)

            # distance features
           for lm in landmarks:
               dist = math.sqrt((lm.x - base_x)**2 + (lm.y - base_y)**2)
               row.append(dist)
  
           if len(row) != 63:
               return {"gesture": current_gesture, "objects": objects}

           arr = np.array(row).reshape(1, -1)

           probs = gesture_model.predict_proba(arr)[0]
           pred = gesture_model.classes_[np.argmax(probs)]
           confidence = np.max(probs)

           if confidence < 0.6:
               return {"gesture": current_gesture, "objects": objects}

           history.append(pred)
           stable_pred = Counter(history).most_common(1)[0][0]

           current_time = time.time()
           if stable_pred != current_gesture:
               if current_time - last_change_time > hold_time:
                   current_gesture = stable_pred
                   last_change_time = current_time

           return {"gesture": current_gesture, "objects": objects}

       return {"gesture": "none", "objects": objects}

   except Exception as e:
      print("Error:", e)
      return {"gesture": "error", "objects": []}



    # python -m uvicorn src.app.main:app --reload