from pathlib import Path
import shutil
import random

# CONFIG — adjust as you like
ROOT_IN  = Path('/home/hanew/your_project_folder/omniacc/data/data_patches_manhat')  # Path, not str
ROOT_OUT = Path('/home/hanew/your_project_folder/omniacc/data/data_split_manhat_no_road')    # Path, not str
SPLITS   = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED     = 42
COPY     = True

def make_dirs():
    for split in SPLITS:
        (ROOT_OUT / split).mkdir(parents=True, exist_ok=True)

def gather_patches():
    # each subfolder of ROOT_IN is one sample
    return [p for p in ROOT_IN.iterdir() if p.is_dir()]

def split_list(items, fractions):
    n = len(items)
    counts = {k: int(v * n) for k, v in fractions.items()}
    leftover = n - sum(counts.values())
    counts["train"] += leftover
    splits = {}
    idx = 0
    for k in ["train", "val", "test"]:
        cnt = counts[k]
        splits[k] = items[idx: idx + cnt]
        idx += cnt
    return splits

def copy_patches(splits):
    for split, folders in splits.items():
        for folder in folders:
            dest = ROOT_OUT / split / folder.name
            if COPY:
                # If you plan to re-run, consider dirs_exist_ok=True (Py3.8+)
                shutil.copytree(folder, dest)
            else:
                shutil.move(str(folder), str(dest))

def main():
    random.seed(SEED)
    make_dirs()
    patches = gather_patches()
    random.shuffle(patches)
    splits = split_list(patches, SPLITS)
    for k, v in splits.items():
        print(f"{k}: {len(v)} samples")
    copy_patches(splits)
    print("Done!")

if __name__ == "__main__":
    main()
