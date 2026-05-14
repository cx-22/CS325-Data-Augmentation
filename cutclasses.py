import os
import shutil

ROOT_DIR = "archive"

allowed = {
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
}

def clean_split(split):
    split_path = os.path.join(ROOT_DIR, split)
    
    if not os.path.exists(split_path):
        return

    for folder in os.listdir(split_path):
        folder_path = os.path.join(split_path, folder)

        if os.path.isdir(folder_path):
            if folder not in allowed:
                shutil.rmtree(folder_path)

clean_split("train")
clean_split("test")

print("Done.")