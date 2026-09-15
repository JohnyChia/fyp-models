# FYP Durian Detection Models

This repository contains the datasets, training code, evaluation results, and model assets for a final-year project on computer-vision-based durian analysis. The project uses Ultralytics YOLO models to detect durian fruit, identify durian trees, and classify diseases on durian leaves.

## Project tasks

- **Durian detection:** detects five durian classes (`d101`, `d175`, `d197`, `d2`, and `d24`).
- **Durian leaf disease detection:** detects Algal, Blight, Colletotrichum, Healthy, Phomopsis, and Rhizoctonia leaves.
- **Durian tree detection:** detects durian trees in images.
- **Model comparison:** compares YOLOv8, YOLO11, and YOLO26 experiments using precision, recall, mAP50, and mAP50-95.

## Repository structure

```text
.
|-- codes/                       # Dataset validation, auditing, and chart utilities
|-- fyp_datasets/
|   |-- durian/                  # Durian detection dataset and data.yaml
|   |-- durian_leaf/             # Leaf disease dataset and data.yaml
|   `-- durian_tree/             # Tree detection dataset and data.yaml
|-- train_models/
|   |-- codes/                   # Training, testing, inference, and comparison scripts
|   |-- scripts/train_all.py     # Runs the model/dataset experiment matrix
|   |-- results/diagram/         # Generated comparison charts
|   |-- runs/                    # Training and evaluation artifacts
|   `-- baselines/               # Baseline experiment artifacts and exported weights
|-- runs/                        # Final training and test outputs
|-- yolo11n.pt                   # YOLO11 pretrained weights
|-- yolo26n.pt                   # YOLO26 pretrained weights
`-- yolov8n.pt                   # YOLOv8 pretrained weights
```

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with CUDA is recommended for training
- Ultralytics YOLO and PyTorch

Install the Python packages used by the project:

```powershell
python -m pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib pyyaml ensemble-boxes
```

## Dataset configuration

Each dataset has an Ultralytics configuration file at `fyp_datasets/<dataset>/data.yaml`. The training scripts currently use the Windows project path:

```text
C:\fyp basic models
```

If the repository is stored elsewhere, update `BASE_DIR` and any absolute paths in the scripts before running them.

## Training

To run the full matrix of three datasets and three YOLO variants, review the device, batch size, epoch count, and weight paths in `train_models/scripts/train_all.py`, then run:

```powershell
python train_models/scripts/train_all.py
```

To train and evaluate the YOLO11 configuration used by the main experiment script:

```powershell
python train_models/codes/run.py
```

The default experiment settings are 150 epochs, 640-pixel images, and CUDA device `0`. Adjust these constants in the selected script to match the available hardware.

## Evaluation and analysis

- `train_models/codes/run_test.py` evaluates trained models on the configured test splits.
- `train_models/codes/compare_models.py` creates model-comparison visualizations.
- `train_models/codes/train_tree.py` performs tree and leaf ensemble inference using weighted box fusion.
- `codes/dataset_audit.py` and `codes/final_integrity_check.py` audit dataset structure and labels.
- `codes/chart.py` and the comparison scripts generate charts in `train_models/results/diagram/`.

Run a utility from the repository root, for example:

```powershell
python codes/final_integrity_check.py
python train_models/codes/compare_models.py
```

## Results

Training runs include model weights, metrics, confusion matrices, prediction samples, and performance curves. Summary visualizations are stored in `train_models/results/diagram/`, while detailed Ultralytics outputs are stored under `runs/` and `train_models/runs/`.

Model checkpoints and datasets make this repository large. Clone it with a stable connection and ensure sufficient disk space is available.
