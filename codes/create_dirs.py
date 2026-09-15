import os
import shutil

BASE_DIR = "C:/fyp basic models"

DIRS_TO_CREATE = [
    "archive",
    "archive/old_datasets",
    "archive/old_models",
    "archive/old_runs",
    "archive/old_scripts",
    "evaluation",
    "evaluation/durian",
    "evaluation/durian_leaf",
    "evaluation/durian_tree",
    "export_models",
    "export_models/durian",
    "export_models/durian_leaf",
    "export_models/durian_tree",
    "export_models/durian/YOLOv8",
    "export_models/durian/YOLO11",
    "export_models/durian/YOLO26",
    "export_models/durian_leaf/YOLOv8",
    "export_models/durian_leaf/YOLO11",
    "export_models/durian_leaf/YOLO26",
    "export_models/durian_tree/YOLOv8",
    "export_models/durian_tree/YOLO11",
    "export_models/durian_tree/YOLO26",
    "fyp_datasets/durian",
    "fyp_datasets/durian_leaf",
    "fyp_datasets/durian_tree",
    "train_models/configs",
    "train_models/scripts",
    "train_models/baselines",
    "train_models/pretrained",
    "results",
    "results/durian/YOLOv8",
    "results/durian/YOLO11",
    "results/durian/YOLO26",
    "results/durian_leaf/YOLOv8",
    "results/durian_leaf/YOLO11",
    "results/durian_leaf/YOLO26",
    "results/durian_tree/YOLOv8",
    "results/durian_tree/YOLO11",
    "results/durian_tree/YOLO26",
    "codes"
]

for dataset in ["durian", "durian_leaf", "durian_tree"]:
    for split in ["train", "val", "test"]:
        DIRS_TO_CREATE.append(f"fyp_datasets/{dataset}/{split}/images")
        DIRS_TO_CREATE.append(f"fyp_datasets/{dataset}/{split}/labels")
    DIRS_TO_CREATE.append(f"fyp_datasets/{dataset}/audit")
    DIRS_TO_CREATE.append(f"fyp_datasets/{dataset}/excluded")

for d in DIRS_TO_CREATE:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

print("Directories created successfully.")
