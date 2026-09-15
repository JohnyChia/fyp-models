import os

BASE_DIR = r"C:\fyp basic models"

DATASETS = [
    "durian",
    "durian_leaf",
    "durian_tree"
]

MODELS = [
    "YOLOv8",
    "YOLO11",
    "YOLO26"
]

MODEL_WEIGHTS = {
    "YOLOv8": r"train_models\pretrained\yolov8n.pt",
    "YOLO11": r"train_models\pretrained\yolo11n.pt",
    "YOLO26": r"train_models\pretrained\yolo26n.pt"
}

EPOCHS = 150
IMGSZ = 640
BATCH = 16
DEVICE = "0"

print("Ready to start 9 YOLO training experiments.")

for ds in DATASETS:

    for model in MODELS:

        print("=" * 70)
        print(f"Experiment: {model} on {ds} dataset")
        print("=" * 70)

        yaml_config = os.path.join(
            BASE_DIR,
            "fyp_datasets",
            ds,
            "data.yaml"
        )

        weights = os.path.join(
            BASE_DIR,
            MODEL_WEIGHTS[model]
        )

        project_dir = os.path.join(
            BASE_DIR,
            "train_models",
            "final",
            ds
        )

        name = model

        command = (
            f'yolo detect train '
            f'data="{yaml_config}" '
            f'model="{weights}" '
            f'epochs={EPOCHS} '
            f'imgsz={IMGSZ} '
            f'batch={BATCH} '
            f'device={DEVICE} '
            f'project="{project_dir}" '
            f'name="{name}" '
            f'exist_ok=True'
        )

        print(f"Command:\n{command}")

        os.system(command)

        print("-" * 70)