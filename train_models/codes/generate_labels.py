from ultralytics import YOLO
import os
import shutil

model = YOLO('yolov8x.pt')
source_root = r'C:\fyp basic models\train\images\ExDark'
save_lbl_dir = r'C:\fyp basic models\train\labels'
temp_labels_dir = r'C:\fyp basic models\runs\auto_labels\labels'

def run_labeling():
    print("Starting AI-powered automatic annotation...")
    for category in os.listdir(source_root):
        cat_path = os.path.join(source_root, category)
        if os.path.isdir(cat_path):
            print(f"Annotating category: {category}")
            model.predict(
                source=cat_path, 
                save_txt=True, 
                project=r'C:\fyp basic models\runs', 
                name='auto_labels', 
                exist_ok=True
            )
            
def sync_labels():
    for category in os.listdir(source_root):
        cat_path = os.path.join(source_root, category)
        if not os.path.isdir(cat_path): continue
        for img_name in os.listdir(cat_path):
            file_base = os.path.splitext(img_name)[0]
            txt_name = file_base + ".txt"
            src_txt = os.path.join(temp_labels_dir, txt_name)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, os.path.join(save_lbl_dir, f"{category}_{txt_name}"))

if __name__ == "__main__":
    run_labeling()
    sync_labels()