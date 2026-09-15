import os
import glob
import yaml

BASE_DIR = "C:/fyp basic models"
DATASETS_DIR = os.path.join(BASE_DIR, "fyp_datasets")

def get_class_count(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path): return None, []
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    nc = data.get("nc", len(data.get("names", [])))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())
    return nc, names

def validate_split(dataset_dir, split, class_count):
    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return {"images": 0, "labels": 0, "invalid": 0, "missing_label": 0, "missing_image": 0, "stems": set()}

    images = glob.glob(os.path.join(images_dir, "*.*"))
    labels = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    image_stems = {os.path.splitext(os.path.basename(img))[0] for img in images}
    label_stems = {os.path.splitext(os.path.basename(lbl))[0] for lbl in labels}
    
    missing_labels = len(image_stems - label_stems)
    missing_images = len(label_stems - image_stems)
    
    common_stems = image_stems.intersection(label_stems)
    
    invalid = 0
    for stem in common_stems:
        lbl_path = os.path.join(labels_dir, f"{stem}.txt")
        is_valid = True
        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        is_valid = False; break
                    cls_id = int(parts[0])
                    if cls_id < 0 or (class_count is not None and cls_id >= class_count):
                        is_valid = False; break
                    for i in range(1, 5):
                        val = float(parts[i])
                        if val < 0 or val > 1:
                            is_valid = False; break
        except Exception:
            is_valid = False
        if not is_valid: invalid += 1

    return {
        "images": len(images), 
        "labels": len(labels), 
        "invalid": invalid, 
        "missing_label": missing_labels, 
        "missing_image": missing_images, 
        "stems": image_stems
    }

for dataset in ["durian", "durian_leaf", "durian_tree"]:
    print(f"===== INTEGRITY CHECK: {dataset} =====")
    dataset_dir = os.path.join(DATASETS_DIR, dataset)
    class_count, class_names = get_class_count(dataset_dir)
    
    train_stats = validate_split(dataset_dir, "train", class_count)
    val_stats = validate_split(dataset_dir, "val", class_count)
    test_stats = validate_split(dataset_dir, "test", class_count)
    
    train_stems = train_stats["stems"]
    val_stems = val_stats["stems"]
    test_stems = test_stats["stems"]
    
    train_val_overlap = len(train_stems.intersection(val_stems))
    train_test_overlap = len(train_stems.intersection(test_stems))
    val_test_overlap = len(val_stems.intersection(test_stems))
    
    print(f"- train/images count: {train_stats['images']}")
    print(f"- train/labels count: {train_stats['labels']}")
    print(f"- val/images count: {val_stats['images']}")
    print(f"- val/labels count: {val_stats['labels']}")
    print(f"- test/images count: {test_stats['images']}")
    print(f"- test/labels count: {test_stats['labels']}")
    
    total_invalid = train_stats['invalid'] + val_stats['invalid'] + test_stats['invalid']
    total_missing_labels = train_stats['missing_label'] + val_stats['missing_label'] + test_stats['missing_label']
    total_missing_images = train_stats['missing_image'] + val_stats['missing_image'] + test_stats['missing_image']
    
    print(f"- invalid annotation rows: {total_invalid}")
    print(f"- images without labels: {total_missing_labels}")
    print(f"- labels without images: {total_missing_images}")
    print(f"- class distribution: {class_names}")
    
    print(f"- train intersection val = {'empty' if train_val_overlap == 0 else str(train_val_overlap) + ' overlaps'}")
    print(f"- train intersection test = {'empty' if train_test_overlap == 0 else str(train_test_overlap) + ' overlaps'}")
    print(f"- val intersection test = {'empty' if val_test_overlap == 0 else str(val_test_overlap) + ' overlaps'}")
    print("\n")
