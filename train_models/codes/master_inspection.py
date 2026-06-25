import cv2
import pandas as pd
import os
import yaml
from ultralytics import YOLO

BASE_DIR = r'C:\fyp basic models'
DATA_YAML = os.path.join(BASE_DIR, 'object.yaml')

def load_names_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

LEAF_NAMES = load_names_from_yaml(DATA_YAML)

model_tree = YOLO(os.path.join(BASE_DIR, r'runs\detect\durian_tree\v26\weights\best.pt'))
model_leaf_v8 = YOLO(os.path.join(BASE_DIR, r'runs\detect\durian_leaf\v8\weights\best.pt'))
model_leaf_v26 = YOLO(os.path.join(BASE_DIR, r'runs\detect\durian_leaf\v26\weights\best.pt'))
model_fruit = YOLO(os.path.join(BASE_DIR, r'runs\detect\durian\v26\weights\best.pt'))

cap = cv2.VideoCapture(r'C:\Users\Johny\Downloads\durian.mp4')
tree_archives = {} 
frame_count = 0

print("tarting industrial-grade integrated inspection: Tree, Durian, and Disease detection mode...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1
    results = model_tree.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    res = results[0]
    
    if res.boxes.id is not None:
        ids = res.boxes.id.cpu().numpy().astype(int)
        boxes = res.boxes.xyxy.cpu().numpy()
        
        for tree_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = map(int, box)
            
            if tree_id not in tree_archives:
                tree_archives[tree_id] = {"diseases": {}, "fruit_count": 0, "total_detections": 0}
            tree_archives[tree_id]["total_detections"] += 1
 
            if frame_count % 10 == 0:
                pad = int((x2 - x1) * 0.2)
                crop = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), 
                             max(0, x1-pad):min(frame.shape[1], x2+pad)]
                
                if crop.size > 0:
                    res_leafs = [model_leaf_v8.predict(crop, conf=0.01, verbose=False)[0],
                                 model_leaf_v26.predict(crop, conf=0.01, verbose=False)[0]]
                    res_f = model_fruit.predict(crop, conf=0.3, verbose=False)[0]

                    tree_archives[tree_id]["fruit_count"] = len(res_f.boxes)
                    for res_l in res_leafs:
                        for l_box in res_l.boxes:
                            label = LEAF_NAMES.get(int(l_box.cls.item()), "Unknown")
                            if label != "Leaf_Healthy":
                                if label not in tree_archives[tree_id]["diseases"]:
                                    tree_archives[tree_id]["diseases"][label] = []
                                tree_archives[tree_id]["diseases"][label].append(float(l_box.conf.item()))
                    
                    for f_box in res_f.boxes:
                        fx1, fy1, fx2, fy2 = map(int, f_box.xyxy[0].cpu().numpy())
                        gx1, gy1 = max(0, x1 - pad + fx1), max(0, y1 - pad + fy1)
                        cv2.rectangle(frame, (gx1, gy1), (gx1 + (fx2-fx1), gy1 + (fy2-fy1)), (0, 255, 0), 2)
                        cv2.putText(frame, "Durian", (gx1, gy1-5), 0, 0.4, (0, 255, 0), 2)

                    for res_l in res_leafs:
                        for l_box in res_l.boxes:
                            cls_id = int(l_box.cls.item())
                            label = LEAF_NAMES.get(cls_id, "Unknown")
                            if label != "Leaf_Healthy":
                                lx1, ly1, lx2, ly2 = map(int, l_box.xyxy[0].cpu().numpy())
                                gx1, gy1 = max(0, x1 - pad + lx1), max(0, y1 - pad + ly1)
                                cv2.rectangle(frame, (gx1, gy1), (gx1 + (lx2-lx1), gy1 + (ly2-ly1)), (0, 0, 255), 2)
                                cv2.putText(frame, label.replace("Leaf_", ""), (gx1, gy1-5), 0, 0.4, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Tree:{tree_id}", (x1, y1-10), 0, 0.5, (255, 0, 0), 2)

    cv2.imshow("Master Inspection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

df_data = []
for tid, info in tree_archives.items():
    if info["total_detections"] >= 10:
        diseases = info["diseases"]
        primary = max(diseases, key=lambda k: len(diseases[k])) if diseases else "Leaf_Healthy"
        avg_conf = sum(diseases[primary]) / len(diseases[primary]) if diseases else 1.0
        df_data.append({"Tree_ID": tid, "Durian_Count": info["fruit_count"], "Disease_Type": primary, "Confidence": round(avg_conf, 3), "Detections": info["total_detections"]})

pd.DataFrame(df_data).to_csv(os.path.join(BASE_DIR, 'results/csv/master_report.csv'), index=False)
print("Report has been generated and saved to master_report.csv")