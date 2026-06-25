import cv2
import numpy as np
from ultralytics import YOLO

MODEL_V8_PATH = r'C:\fyp basic models\runs\detect\durian\v8\weights\best.pt'
MODEL_V26_PATH = r'C:\fyp basic models\runs\detect\durian\v26\weights\best.pt'
VIDEO_PATH = r'C:\Users\Johny\Downloads\durian_fruits.mp4'

model_v8 = YOLO(MODEL_V8_PATH)
model_v26 = YOLO(MODEL_V26_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

all_confidences = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    res_v8 = model_v8.predict(frame, conf=0.01, verbose=False)[0]
    res_v26 = model_v26.predict(frame, conf=0.01, verbose=False)[0]
    
    for res in [res_v8, res_v26]:
        if res.boxes is not None:
            for box in res.boxes:
                all_confidences.append(box.conf.item())

cap.release()

if all_confidences:
    all_confidences = np.array(all_confidences)

    print("\n" + "=" * 40)
    print("=" * 40)
    print(f"Total detected bounding boxes (including noise): {len(all_confidences)}")
    print(f"Maximum Confidence (Max Conf): {all_confidences.max():.4f}")
    print(f"Average Confidence (Mean Conf): {all_confidences.mean():.4f}")
    print(f"Median Confidence (Median Conf): {np.median(all_confidences):.4f}")
    print(f"Top 10% Confidence Threshold (90th Percentile): {np.percentile(all_confidences, 90):.4f}")
    print(f"Top 5% Confidence Threshold (95th Percentile): {np.percentile(all_confidences, 95):.4f}")
    print("=" * 40)
else:
    print("No objects were detected. Please check the model or video path.")