import os
import shutil

import cv2
from tqdm import tqdm
from ultralytics import YOLO


class MITScenesProcessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.model = YOLO('yolov8n.pt')
        pass

    def process(self):
        image_extensions = ('.jpg', '.jpeg', '.png')
        peopleless = os.path.join(self.dataset_dir, "Other")
        for root, _, files in os.walk(self.dataset_dir):
            for file in tqdm(files, desc="Processing images"):
                if file.lower().endswith(image_extensions):
                    try:
                        image_path = os.path.join(root, file)
                        img = cv2.imread(image_path)

                        results = self.model(img, verbose=False)
                        detected_classes = [self.model.model.names[int(cls)] for cls in results[0].boxes.cls]

                        if "person" not in detected_classes:
                            save_path = os.path.join(peopleless, file)
                            cv2.imwrite(save_path, img)
                    except Exception as e:
                        print(f"Error processing {file}", e)