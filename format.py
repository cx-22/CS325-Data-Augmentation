import os
import random
import shutil
from PIL import Image

random.seed(42)

ANIMALS = [
    "Zebra",
    "Lion",
    "Leopard",
    "Cheetah",
    "Bear",
    "Brown bear",
    "Canary",
    "Crocodile",
    "Polar bear",
    "Bull",
    "Camel",
    "Cattle",
    "Sheep",
    "Rhinoceros",
    "Mule",
    "Elephant",
    "Fox"
]

CLASS_MAP = {name: i for i, name in enumerate(ANIMALS)}


def convert_label(line, img_w, img_h):
    line = line.strip()
    if not line:
        return None

    parts = line.split()

    if len(parts) < 5:
        return None

    x1, y1, x2, y2 = map(float, parts[-4:])

    cls_name = " ".join(parts[:-4])

    if cls_name not in CLASS_MAP:
        raise ValueError(f"Unknown class name: '{cls_name}'")

    class_id = CLASS_MAP[cls_name]

    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return f"{class_id} {xc} {yc} {w} {h}\n"




def balance_samples(samples, max_per_class):
    per_class = {}

    for img_path, lbl_path, animal in samples:
        if animal not in per_class:
            per_class[animal] = []
        per_class[animal].append((img_path, lbl_path, animal))

    balanced = []

    for animal, items in per_class.items():
        if len(items) < max_per_class:
            selected = items
        else:
            random.shuffle(items)
            selected = items[:max_per_class]

        balanced.extend(selected)

    return balanced


def split_per_class(samples, train_ratio=0.8):
    per_class = {}

    for s in samples:
        animal = s[2]
        if animal not in per_class:
            per_class[animal] = []
        per_class[animal].append(s)

    train_set = []
    val_set = []

    for animal, items in per_class.items():
        split_idx = int(len(items) * train_ratio)
        train_set.extend(items[:split_idx])
        val_set.extend(items[split_idx:])

    return train_set, val_set




def ensure_dirs(dst):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dst, "images", split), exist_ok=True)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dst, "labels", split), exist_ok=True)


def load_samples(root):
    samples = []

    if not os.path.exists(root):
        raise FileNotFoundError(f"Missing path: {root}")

    for animal in os.listdir(root):
        animal_dir = os.path.join(root, animal)

        if not os.path.isdir(animal_dir):
            continue

        if animal not in CLASS_MAP:
            continue

        img_dir = animal_dir
        lbl_dir = os.path.join(animal_dir, "Label")

        if not os.path.exists(lbl_dir):
            print(f"[WARN] Missing label folder: {lbl_dir}")
            continue

        for f in os.listdir(img_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_dir, f)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")

            if not os.path.exists(lbl_path):
                print(f"[WARN] Missing label file: {lbl_path}")
                continue

            samples.append((img_path, lbl_path, animal))

    return samples


def process_samples(samples, dst, split):
    image_paths = []

    for img_path, lbl_path, animal in samples:
        name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{animal.replace(' ', '_')}_{name}"

        rel_path = f"images/{split}/{new_name}.jpg"
        image_paths.append(rel_path)

        out_img = os.path.join(dst, rel_path)
        shutil.copy2(img_path, out_img)

        with Image.open(img_path) as img:
            w, h = img.size

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        out_lines = []
        for line in lines:
            converted = convert_label(line, w, h)
            if converted is not None:
                out_lines.append(converted)

        if split != "test":
            out_lbl = os.path.join(dst, "labels", split, new_name + ".txt")
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.writelines(out_lines)

    return image_paths

def process_test(root, dst):
    samples = load_samples(root)

    for img_path, lbl_path, animal in samples:
        name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{animal.replace(' ', '_')}_{name}"

        out_img = os.path.join(dst, "images", "test", new_name + ".jpg")
        shutil.copy2(img_path, out_img)

        with Image.open(img_path) as img:
            w, h = img.size

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        out_lines = []
        for line in lines:
            converted = convert_label(line, w, h)
            if converted is not None:
                out_lines.append(converted)


def main():
    src_train = "archive/train"
    src_test = "archive/test"
    dst = "animals"

    ensure_dirs(dst)

    train_samples = load_samples(src_train)

    train_samples = balance_samples(train_samples, 47)

    train_set, val_set = split_per_class(train_samples, 0.8)

    train_paths = process_samples(train_set, dst, "train")
    val_paths = process_samples(val_set, dst, "val")

    with open(os.path.join(dst, "train.txt"), "w") as f:
        for p in train_paths:
            f.write(p + "\n")

    with open(os.path.join(dst, "val.txt"), "w") as f:
        for p in val_paths:
            f.write(p + "\n")

    test_samples = load_samples(src_test)

    test_paths = process_samples(test_samples, dst, "test")

    with open(os.path.join(dst, "test.txt"), "w") as f:
        for p in test_paths:
            f.write(p + "\n")

if __name__ == "__main__":
    main()