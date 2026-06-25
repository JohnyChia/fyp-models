import cv2
from ultralytics import YOLO
import os

model = YOLO('C:/fyp basic models/runs/detect/durian_tree/v26/weights/best.pt') 
video_path = 'C:/Users/Johny/Downloads/durian_fruits.mp4'
output_dir = 'dataset_auto_labels'
os.makedirs(f"{output_dir}/labels", exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    if frame_idx % 30 == 0:
        results = model.predict(frame, conf=0.5) 
        img_name = f"frame_{frame_idx}.jpg"
        cv2.imwrite(f"{output_dir}/images/{img_name}", frame)
   
        with open(f"{output_dir}/labels/frame_{frame_idx}.txt", "w") as f:
            for box in results[0].boxes:
                x, y, w, h = box.xywhn[0]
                cls = int(box.cls[0])
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
    frame_idx += 1

cap.release()