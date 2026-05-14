import os
import cv2
import albumentations as A
from pathlib import Path
AUGMENTATIONS = [
    A.HorizontalFlip(p=0.5),  
    A.MotionBlur(blur_range=(8, 12), p=1.0),
    A.AtmosphericFog(density_range=(1.0, 2.5), depth_mode="linear", p=1.0),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
]
def load_yolo_labels(label_path):
    boxes, class_labels = [], []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_labels.append(parts[0])
                boxes.append([float(x) for x in parts[1:]])
    return boxes, class_labels

def save_yolo_labels(label_path, boxes, class_labels):
    with open(label_path, 'w') as f:
        for cls, box in zip(class_labels, boxes):
            f.write(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

def augment_dataset(images_dir, labels_dir):
    image_files = list(Path(images_dir).glob("*.jpg"))
    for img_path in image_files:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, class_labels = load_yolo_labels(label_path)
        if not boxes:
            continue
        for i, aug in enumerate(AUGMENTATIONS):
            transform = A.Compose(
                [aug],
                bbox_params=A.BboxParams(
                    coord_format='yolo',
                    label_fields=['class_labels'],
                    min_visibility=0.3
                )
            )
            try:
                result = transform(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
            except Exception as e:
                continue
            if len(result['bboxes']) == 0:
                continue
            new_img_name = f"{img_path.stem}_aug{i}.jpg"
            new_img_path = Path(images_dir) / new_img_name
            aug_img = cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), aug_img)
            new_lbl_path = Path(labels_dir) / f"{img_path.stem}_aug{i}.txt"
            save_yolo_labels(str(new_lbl_path), result['bboxes'], result['class_labels'])
augment_dataset(
    "animals/images/train",
    "animals/labels/train"
)