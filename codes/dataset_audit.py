import os
import shutil
import glob
import yaml

BASE_DIR = "C:/fyp basic models"
DATASETS_DIR = os.path.join(BASE_DIR, "fyp_datasets")

def get_class_count(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("nc", len(data.get("names", [])))

def is_valid_label(label_path, class_count):
    if not os.path.exists(label_path):
        return False
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                cls_id = int(parts[0])
                if cls_id < 0 or (class_count is not None and cls_id >= class_count):
                    return False
                for i in range(1, 5):
                    val = float(parts[i])
                    if val < 0 or val > 1:
                        return False
        return True
    except Exception:
        return False

def audit_split(dataset_dir, split, class_count, audit_dir):
    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return {"images": 0, "labels": 0, "invalid": 0, "missing_label": 0, "missing_image": 0}

    images = glob.glob(os.path.join(images_dir, "*.*"))
    labels = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    image_stems = {os.path.splitext(os.path.basename(img))[0]: img for img in images}
    label_stems = {os.path.splitext(os.path.basename(lbl))[0]: lbl for lbl in labels}
    
    stats = {"images": 0, "labels": 0, "invalid": 0, "missing_label": 0, "missing_image": 0}
    
    for stem, img_path in list(image_stems.items()):
        if stem not in label_stems:
            shutil.move(img_path, os.path.join(audit_dir, os.path.basename(img_path)))
            stats["missing_label"] += 1
            del image_stems[stem]
        else:
            lbl_path = label_stems[stem]
            if not is_valid_label(lbl_path, class_count):
                shutil.move(img_path, os.path.join(audit_dir, os.path.basename(img_path)))
                shutil.move(lbl_path, os.path.join(audit_dir, os.path.basename(lbl_path)))
                stats["invalid"] += 1
                del image_stems[stem]
                del label_stems[stem]
            else:
                stats["images"] += 1
                stats["labels"] += 1
    
    for stem, lbl_path in list(label_stems.items()):
        if stem not in image_stems:
            shutil.move(lbl_path, os.path.join(audit_dir, os.path.basename(lbl_path)))
            stats["missing_image"] += 1
            del label_stems[stem]

    return stats

def audit_dataset(dataset_name):
    print(f"===== DATASET: {dataset_name} =====")
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    audit_dir = os.path.join(dataset_dir, "audit")
    
    os.makedirs(audit_dir, exist_ok=True)
    
    class_count = get_class_count(dataset_dir)
    print(f"Class count from data.yaml: {class_count}")
    
    total_stats = {"images": 0, "labels": 0, "invalid": 0, "missing_label": 0, "missing_image": 0}
    
    for split in ["train", "val", "test"]:
        stats = audit_split(dataset_dir, split, class_count, audit_dir)
        print(f"Split {split}: {stats['images']} valid images/labels")
        for k in total_stats:
            total_stats[k] += stats[k]
            
    print(f"Total valid images: {total_stats['images']}")
    print(f"Total valid labels: {total_stats['labels']}")
    print(f"Total invalid annotations: {total_stats['invalid']}")
    print(f"Images without labels: {total_stats['missing_label']}")
    print(f"Labels without images: {total_stats['missing_image']}")
    print("===================================\n")

if __name__ == "__main__":
    for ds in ["durian", "durian_leaf", "durian_tree"]:
        audit_dataset(ds)
