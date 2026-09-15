from ultralytics import YOLO
from pathlib import Path
import torch

BASE_DIR = Path(r"C:\fyp basic models")

DATASETS = [
    "durian",
    "durian_leaf",
    "durian_tree",
]

MODEL_NAME = "YOLO11"
MODEL_WEIGHT = "yolo11n.pt"

EPOCHS = 150
IMGSZ = 640
BATCH = 4
DEVICE = 0
WORKERS = 0

TRAIN_ROOT = (
    BASE_DIR
    / "runs"
    / "detect"
    / "train_models"
    / "final"
)

TEST_ROOT = (
    BASE_DIR
    / "runs"
    / "detect"
    / "test_evaluation"
)

TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
TEST_ROOT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("YOLO11 TRAINING + TEST EVALUATION")
print("=" * 70)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is NOT available!")

print("=" * 70)

for dataset in DATASETS:

    print("\n")
    print("=" * 70)
    print(f"DATASET: {dataset}")
    print("=" * 70)

    data_yaml = (
        BASE_DIR
        / "fyp_datasets"
        / dataset
        / "data.yaml"
    )

    if not data_yaml.exists():
        print("ERROR: data.yaml not found:")
        print(data_yaml)
        continue

    train_name = f"{dataset}_yolo11"

    train_dir = TRAIN_ROOT / train_name

    print(f"Dataset       : {dataset}")
    print(f"Model         : {MODEL_NAME}")
    print(f"Weights       : {MODEL_WEIGHT}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Image size    : {IMGSZ}")
    print(f"Batch         : {BATCH}")
    print(f"Workers       : {WORKERS}")
    print(f"Device        : {DEVICE}")
    print(f"Data YAML     : {data_yaml}")
    print(f"Train output  : {train_dir}")

    print("=" * 70)

    try:

        print("\nStarting training...")

        model = YOLO(MODEL_WEIGHT)

        model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,

            project=str(TRAIN_ROOT),
            name=train_name,
            exist_ok=True,

            pretrained=True,
            amp=True,

            save=True,
            plots=True,
            verbose=True,
        )

        print("\n")
        print("=" * 70)
        print(f"TRAINING FINISHED: {dataset} - YOLO11")
        print("=" * 70)

    except Exception as e:

        print("\n")
        print("=" * 70)
        print(f"TRAINING FAILED: {dataset} - YOLO11")
        print("=" * 70)

        print(f"Error type : {type(e).__name__}")
        print(f"Error      : {e}")

        continue

    best_model = train_dir / "weights" / "best.pt"

    if not best_model.exists():

        print("\nERROR: best.pt not found:")
        print(best_model)

        continue

    print("\n")
    print("=" * 70)
    print("BEST MODEL FOUND")
    print("=" * 70)

    print(best_model)

    test_name = train_name

    test_dir = TEST_ROOT / test_name

    print("\n")
    print("=" * 70)
    print(f"TEST EVALUATION: {dataset} - YOLO11")
    print("=" * 70)

    print(f"Model : {best_model}")
    print(f"Data  : {data_yaml}")
    print(f"Save  : {test_dir}")

    try:

        print("\nStarting test evaluation...")

        test_model = YOLO(str(best_model))

        metrics = test_model.val(
            data=str(data_yaml),

            split="test",

            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,

            project=str(TEST_ROOT),
            name=test_name,
            exist_ok=True,

            plots=True,
            verbose=True,
        )

        print("\n")
        print("=" * 70)
        print(f"TEST FINISHED: {dataset} - YOLO11")
        print("=" * 70)

        try:

            print(f"mAP50     : {metrics.box.map50:.4f}")
            print(f"mAP50-95  : {metrics.box.map:.4f}")
            print(f"Precision : {metrics.box.mp:.4f}")
            print(f"Recall    : {metrics.box.mr:.4f}")

        except Exception:

            print("Metrics completed.")
            print("Please check results.csv and generated plots.")

    except Exception as e:

        print("\n")
        print("=" * 70)
        print(f"TEST FAILED: {dataset} - YOLO11")
        print("=" * 70)

        print(f"Error type : {type(e).__name__}")
        print(f"Error      : {e}")

        continue

print("\n")
print("=" * 70)
print("ALL 3 YOLO11 EXPERIMENTS FINISHED")
print("=" * 70)

print("\nTraining directory:")
print(TRAIN_ROOT)

print("\nTest evaluation directory:")
print(TEST_ROOT)

print("\nExpected YOLO11 models:")

for dataset in DATASETS:

    print(
        TRAIN_ROOT
        / f"{dataset}_yolo11"
        / "weights"
        / "best.pt"
    )

print("\nDone.")