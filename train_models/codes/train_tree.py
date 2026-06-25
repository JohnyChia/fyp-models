import cv2
import os
import glob
import numpy as np
import csv
from ensemble_boxes import *
from ultralytics import YOLO

leaf_models = [YOLO(r'C:/fyp basic models/runs/detect/durian_leaf/v8/weights/best.pt'),
               YOLO(r'C:/fyp basic models/runs/detect/durian_leaf/v26/weights/best.pt')]
tree_models = [YOLO(r'C:/fyp basic models/runs/detect/durian_tree/v8/weights/best.pt'),
               YOLO(r'C:/fyp basic models/runs/detect/durian_tree/v26/weights/best.pt')]

CLASS_NAMES = {0: "Algal", 1: "Blight", 2: "Colletotrichum", 3: "Healthy", 4: "Phomopsis", 5: "Rhizoctonia"}

def get_ensemble_predictions(models, img, imgsz, conf_threshold):
    boxes_list, scores_list, labels_list = [], [], []
    for model in models:
        res = model.predict(img, imgsz=imgsz, conf=conf_threshold, verbose=False)[0]
        boxes, scores, labels = [], [], []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([max(0.0, min(1.0, x1 / imgsz)), max(0.0, min(1.0, y1 / imgsz)), 
                          max(0.0, min(1.0, x2 / imgsz)), max(0.0, min(1.0, y2 / imgsz))])
            scores.append(box.conf.item())
            labels.append(int(box.cls.item()))
        boxes_list.append(boxes); scores_list.append(scores); labels_list.append(labels)
    
    if not any(boxes_list): return [], [], []
    return weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=[1, 2], iou_thr=0.7, skip_box_thr=0.001)

def analyze_system(image_path, output_image_path, csv_writer):
    img = cv2.imread(image_path)
    if img is None: return
    h_orig, w_orig = img.shape[:2]
    
    t_boxes, _, _ = get_ensemble_predictions(tree_models, img, imgsz=640, conf_threshold=0.3)
    if len(t_boxes) == 0: t_boxes = [[0.0, 0.0, 1.0, 1.0]]
    
    annotated_img = img.copy()
    all_detections = [] 

    for tb in t_boxes:
        x1_t, y1_t = int(tb[0]*w_orig), int(tb[1]*h_orig)
        x2_t, y2_t = int(tb[2]*w_orig), int(tb[3]*h_orig)
        
        if not (tb[0] == 0.0 and tb[1] == 0.0 and tb[2] == 1.0 and tb[3] == 1.0):
            cv2.rectangle(annotated_img, (x1_t, y1_t), (x2_t, y2_t), (255, 0, 0), 4)
        
        crop_t = img[max(0, y1_t):min(h_orig, y2_t), max(0, x1_t):min(w_orig, x2_t)]
        h_c, w_c = crop_t.shape[:2]

        tiles = []
        if h_c >= 320 and w_c >= 320:
            for y in range(0, h_c, 600):
                for x in range(0, w_c, 600):
                    tile = crop_t[y:y+640, x:x+640]
                    if tile.shape[0] >= 100 and tile.shape[1] >= 100:
                        tiles.append((tile, x, y))
        else:
            tiles = [(crop_t, 0, 0)]

        for tile, x_off, y_off in tiles:
            d_boxes, _, d_labels = get_ensemble_predictions(leaf_models, tile, imgsz=640, conf_threshold=0.01)
            for db, dl in zip(d_boxes, d_labels):
                fx1, fy1 = x1_t + x_off + int(db[0]*tile.shape[1]), y1_t + y_off + int(db[1]*tile.shape[0])
                fx2, fy2 = x1_t + x_off + int(db[2]*tile.shape[1]), y1_t + y_off + int(db[3]*tile.shape[0])
                all_detections.append({'label': dl})
                cv2.rectangle(annotated_img, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

    stats = {name: 0 for name in CLASS_NAMES.values()}
    for det in all_detections: stats[CLASS_NAMES.get(det['label'], "Unknown")] += 1
    total = sum(stats.values())
    health = (stats.get("Healthy", 0) / total * 100) if total > 0 else 0

    side_panel = np.ones((h_orig, 400, 3), dtype=np.uint8) * 255
    cv2.putText(side_panel, f"Total: {total}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(side_panel, f"Health: {health:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if health>70 else (0,0,255), 2)
    
    cv2.imwrite(output_image_path, np.hstack((annotated_img, side_panel)))
    csv_writer.writerow([os.path.basename(image_path), f"{health:.1f}"] + [stats[name] for name in CLASS_NAMES.values()])
    print(f"完成: {os.path.basename(image_path)}")

if __name__ == "__main__":
    base_dir = r'C:\fyp basic models\results'
    img_out_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_out_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, 'summary_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Health Index (%)'] + list(CLASS_NAMES.values()))
        
        for img_path in glob.glob(r'C:\fyp basic models\durian_datasets\durian_tree\images\*.jpg'):
            analyze_system(img_path, os.path.join(img_out_dir, os.path.basename(img_path)), writer)