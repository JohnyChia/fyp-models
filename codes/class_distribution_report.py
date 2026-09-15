import os
import glob
import yaml
from collections import defaultdict

BASE_DIR = "C:/fyp basic models"
DATASETS_DIR = os.path.join(BASE_DIR, "fyp_datasets")

def get_class_names(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path): return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names_list = []
        for k in sorted(names.keys()):
            names_list.append(names[k])
        return names_list
    return names

def analyze_split(dataset_dir, split, class_names):
    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return None

    images = glob.glob(os.path.join(images_dir, "*.*"))
    labels = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    annotation_counts = defaultdict(int)
    image_counts = defaultdict(int)
    
    for lbl_path in labels:
        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                classes_in_image = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        cls_id = int(parts[0])
                        annotation_counts[cls_id] += 1
                        classes_in_image.add(cls_id)
                for cls_id in classes_in_image:
                    image_counts[cls_id] += 1
        except Exception:
            pass
            
    return {
        "num_images": len(images),
        "num_labels": len(labels),
        "annotation_counts": annotation_counts,
        "image_counts": image_counts
    }

def print_report():
    for dataset in ["durian", "durian_leaf", "durian_tree"]:
        print(f"==================================================")
        print(f"DATASET: {dataset.upper()}")
        print(f"==================================================")
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        class_names = get_class_names(dataset_dir)
        
        for split in ["train", "val", "test"]:
            stats = analyze_split(dataset_dir, split, class_names)
            if not stats: continue
            
            print(f"\n--- Split: {split.upper()} ---")
            print(f"Number of images: {stats['num_images']}")
            print(f"Number of annotation files: {stats['num_labels']}\n")
            
            print(f"{'Class Name':<20} | {'Annotations':<12} | {'Images w/ Class':<15}")
            print("-" * 55)
            
            for i, name in enumerate(class_names):
                ann_count = stats['annotation_counts'].get(i, 0)
                img_count = stats['image_counts'].get(i, 0)
                print(f"{name:<20} | {ann_count:<12} | {img_count:<15}")
        print("\n")

if __name__ == "__main__":
    print_report()
