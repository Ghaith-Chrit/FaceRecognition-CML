import os

import cv2
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class YTFacesProcessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.faces_dir = os.path.join(self.dataset_dir, "Faces")
        self.face_index = {}

    def process(self):
        os.makedirs(self.faces_dir, exist_ok=True)
        npz_files = [os.path.join(root, file)
                     for root, _, files in os.walk(self.dataset_dir)
                     for file in files if file.endswith(".npz")]

        print(f"Found {len(npz_files)} files. Processing...")

        with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("Processing Files", total=len(npz_files))

            for file_path in npz_files:
                self.process_file(file_path, progress)
                progress.update(task, advance=1)

    def process_file(self, file_path, progress):
        try:
            with np.load(file_path, mmap_mode="r") as data:
                colour_images = data["colorImages"]
                name = os.path.basename(file_path).replace(".npz", "")
                name = "_".join(name.split("_")[:-1])
                name_dir = os.path.join(self.faces_dir, name)
                if os.path.exists(name_dir):
                    return

                os.makedirs(name_dir, exist_ok=True)

                num_frames = colour_images.shape[-1]
                formatted_name = f"{name[:25]:<25}"

                frame_task = progress.add_task(f"[green]Frames {formatted_name}", total=num_frames // 5)

                for i in range(0, num_frames, 5):
                    frame = colour_images[..., i]
                    self.process_image(name_dir, frame, (i // 5) + self.face_index.get(name, 0))
                    progress.update(frame_task, advance=1)

                self.face_index[name] = (num_frames // 5) + self.face_index.get(name, 0)

                progress.remove_task(frame_task)

        except Exception as e:
            print(f"Error processing {file_path}", e)

    def process_image(self, name_dir, colour_image, index):
        face_path = os.path.join(name_dir, f"face_{index}.png")
        if not os.path.exists(face_path):
            cv2.imwrite(face_path, colour_image)