import os
import shutil
from pathlib import Path

src_root = r'C:\USERS\JOHNY\DOWNLOADS\ARCHIVE (1)\DURIAN_LEAF_DISEASES'
dest_images = r'C:\fyp basic models\train\images'
os.makedirs(dest_images, exist_ok=True)

for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.lower().endswith('.jpg'):
            src_file = os.path.join(root, file)
            sub_folder = os.path.basename(root)
            new_name = f"{sub_folder}_{file}"
            dest_file = os.path.join(dest_images, new_name)
 
            shutil.copy2(src_file, dest_file)
            print(f"Organized: {new_name}")
