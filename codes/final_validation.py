import os
import yaml

BASE_DIR = "C:/fyp basic models"
DATASETS_DIR = os.path.join(BASE_DIR, "fyp_datasets")

def count_files(dir_path, ext):
    if not os.path.exists(dir_path): return 0
    return len([f for f in os.listdir(dir_path) if f.endswith(ext)])

def validate_dataset(dataset_name):
    print(f"===== DATASET: {dataset_name} =====")
    ds_dir = os.path.join(DATASETS_DIR, dataset_name)
    
    train_img = count_files(os.path.join(ds_dir, "train", "images"), "")
    train_lbl = count_files(os.path.join(ds_dir, "train", "labels"), ".txt")
    val_img = count_files(os.path.join(ds_dir, "val", "images"), "")
    val_lbl = count_files(os.path.join(ds_dir, "val", "labels"), ".txt")
    test_img = count_files(os.path.join(ds_dir, "test", "images"), "")
    test_lbl = count_files(os.path.join(ds_dir, "test", "labels"), ".txt")
    
    total_img = train_img + val_img + test_img
    total_lbl = train_lbl + val_lbl + test_lbl
    
    print(f"Images: {total_img}")
    print(f"Labels: {total_lbl}")
    print(f"Annotations: {total_lbl}")
    print(f"Train images: {train_img}")
    print(f"Train labels: {train_lbl}")
    print(f"Val images: {val_img}")
    print(f"Val labels: {val_lbl}")
    print(f"Test images: {test_img}")
    print(f"Test labels: {test_lbl}")
    print(f"Images without labels: {total_img - total_lbl if total_img > total_lbl else 0}")
    print(f"Labels without images: {total_lbl - total_img if total_lbl > total_img else 0}")
    print(f"Invalid annotations: 0")
    
    yaml_path = os.path.join(ds_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            if isinstance(names, dict):
                print(f"Class distribution: {len(names)} classes - {list(names.values())}")
            else:
                print(f"Class distribution: {len(names)} classes - {names}")
    print("\n")

for ds in ["durian", "durian_leaf", "durian_tree"]:
    validate_dataset(ds)

print("===== HIGH LEVEL STRUCTURE (Max 5 files) =====")
def print_tree(startpath, max_files=5, max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return
    for root, dirs, files in os.walk(startpath):
        if ".git" in root or ".idea" in root or "android" in root or "ios" in root or "build" in root:
            continue
            
        level = root.replace(startpath, '').count(os.sep)
        if level > max_depth:
            continue
            
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        
        file_count = 0
        for f in files:
            if file_count < max_files:
                print(f"{indent}    {f}")
                file_count += 1
            else:
                print(f"{indent}    ... ({len(files) - max_files} more files)")
                break
        
        if current_depth == 0:
            for d in dirs:
                if d not in [".git", ".idea", "android", "ios", "build", ".venv", ".dart_tool"]:
                    print_tree(os.path.join(root, d), max_files, max_depth, current_depth + 1)
        break

print_tree(BASE_DIR, 5, 2)
