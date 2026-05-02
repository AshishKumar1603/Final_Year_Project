# from ultralytics import YOLO
# import cv2
# from collections import defaultdict, deque

# # 🔥 per-object history (tracking based)
# track_history = defaultdict(lambda: deque(maxlen=5))

# model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     frame = cv2.resize(frame, (640, 480))

#     # 🔥 tracking enabled
#     results = model.track(frame, persist=True, conf=0.4)

#     if results[0].boxes is not None:

#         boxes = results[0].boxes.xyxy
#         ids = results[0].boxes.id
#         classes = results[0].boxes.cls

#         # ⚠️ tracking IDs sometimes None
#         if ids is not None:
#             ids = ids.int().cpu().tolist()
#         else:
#             ids = [0] * len(boxes)

#         for box, track_id, cls in zip(boxes, ids, classes):
#             box = box.cpu().numpy()

#             # 🔥 per-object smoothing
#             history = track_history[track_id]
#             history.append(box)
#             avg_box = sum(history) / len(history)

#             x1, y1, x2, y2 = map(int, avg_box)

#             # 🎯 bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

#             # 🎯 label + id
#             label = model.names[int(cls)]
#             text = f"{label} ID:{track_id}"

#             cv2.putText(frame, text, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#     cv2.imshow("YOLO Detection", frame)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# # python src/detection/yolo_test.py

from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")
track_history = defaultdict(lambda: deque(maxlen=5))

def detect_objects(frame):
    results = model.track(frame, persist=True, conf=0.4)

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

            history = track_history[track_id]
            history.append(box)
            avg_box = sum(history) / len(history)

            label = model.names[int(cls)]
            detected_objects.append(label)

    return detected_objects