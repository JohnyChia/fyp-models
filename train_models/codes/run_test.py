import os
import csv
import yaml
from datetime import datetime
from ultralytics import YOLO

BASE_DIR         = r"C:\fyp basic models"
TRAIN_MODELS_DIR = os.path.join(BASE_DIR, "train_models")
CODES_DIR        = os.path.join(TRAIN_MODELS_DIR, "codes")
DATASETS_YAML    = os.path.join(CODES_DIR, "durian_datasets.yaml")

VERSIONS = ["v8", "v11", "v26"]
CATEGORIES = ["durian", "durian_leaf", "durian_tree"]

def build_flat_yaml(category_config: dict, out_path: str):
    flat = {
        "train": category_config["train"],
        "val":   category_config["val"],
        "test":  category_config["test"],
        "nc":    category_config["nc"],
        "names": category_config["names"],
    }
    with open(out_path, "w") as f:
        yaml.dump(flat, f, allow_unicode=True, default_flow_style=False)

def main():
    with open(DATASETS_YAML, "r") as f:
        master_config = yaml.safe_load(f)

    all_results = {}

    print(f"\n{'='*65}")
    print(f"  YOLO Test Run  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    for category in CATEGORIES:
        all_results[category] = {}
        cat_cfg = master_config[category]
        temp_yaml = os.path.join(CODES_DIR, f"_tmp_{category}.yaml")
        build_flat_yaml(cat_cfg, temp_yaml)

        print(f"\n  Category : {category.upper()}")
        print(f"  Classes  : {cat_cfg['names']}")
        print("-" * 65)

        for version in VERSIONS:
            weight_path = os.path.join(
                TRAIN_MODELS_DIR, "runs", "detect",
                category, version, "train", "weights", "best.pt"
            )

            print(f"\n  [{version}] weight: {weight_path}")

            if not os.path.exists(weight_path):
                print(f"  WARNING  : weights not found, skipping.")
                all_results[category][version] = {"error": "weights not found"}
                continue

            try:
                model = YOLO(weight_path)

                results = model.val(
                    data=temp_yaml,
                    split="test",
                    imgsz=640,
                    conf=0.25,
                    iou=0.60,
                    plots=True,
                    save_json=False,
                    project=os.path.join(
                        TRAIN_MODELS_DIR, "runs", "detect", category, version
                    ),
                    name="test",
                    exist_ok=True,
                    verbose=False,
                )
                try:
                    losses = results.loss
                    box_loss = round(float(losses[0]), 5) if losses is not None and len(losses) > 0 else ""
                    cls_loss = round(float(losses[1]), 5) if losses is not None and len(losses) > 1 else ""
                    dfl_loss = round(float(losses[2]), 5) if losses is not None and len(losses) > 2 else ""
                except Exception:
                    box_loss, cls_loss, dfl_loss = "", "", ""

                metrics = {
                    "metrics/precision(B)": round(float(results.box.mp),    5),
                    "metrics/recall(B)":    round(float(results.box.mr),    5),
                    "metrics/mAP50(B)":     round(float(results.box.map50), 5),
                    "metrics/mAP50-95(B)":  round(float(results.box.map),   5),
                    "val/box_loss":         box_loss,
                    "val/cls_loss":         cls_loss,
                    "val/dfl_loss":         dfl_loss,
                }
                all_results[category][version] = metrics

                print(f"  Precision    : {metrics['metrics/precision(B)']:.5f}")
                print(f"  Recall       : {metrics['metrics/recall(B)']:.5f}")
                print(f"  mAP@50       : {metrics['metrics/mAP50(B)']:.5f}")
                print(f"  mAP@50-95    : {metrics['metrics/mAP50-95(B)']:.5f}")

            except Exception as e:
                print(f"  ERROR      : {e}")
                all_results[category][version] = {"error": str(e)}

        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)

    cols = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    print(f"\n\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Category':<16} {'Version':<8} {'Precision':>10} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10} {'box_loss':>10} {'cls_loss':>10} {'dfl_loss':>10}")
    print("-" * 80)

    for category, versions in all_results.items():
        for version, m in versions.items():
            if "error" in m:
                print(f"{category:<16} {version:<8}  ERROR -- {m['error']}")
            else:
                print(
                    f"{category:<16} {version:<8}"
                    f" {m['metrics/precision(B)']:>10.5f}"
                    f" {m['metrics/recall(B)']:>8.5f}"
                    f" {m['metrics/mAP50(B)']:>8.5f}"
                    f" {m['metrics/mAP50-95(B)']:>10.5f}"
                    f" {str(m['val/box_loss']):>10}"
                    f" {str(m['val/cls_loss']):>10}"
                    f" {str(m['val/dfl_loss']):>10}"
                )

    csv_columns = [
        "category", "version",
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
    ]

    out_csv = os.path.join(TRAIN_MODELS_DIR, "results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for category, versions in all_results.items():
            for version, m in versions.items():
                if "error" in m:
                    row = {"category": category, "version": version, **{c: "ERROR" for c in csv_columns[2:]}}
                else:
                    row = {"category": category, "version": version, **m}
                writer.writerow(row)

    print(f"\n  Results saved : {out_csv}")
    print(f"  Plots saved   : runs/detect/<category>/<version>/test/")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
