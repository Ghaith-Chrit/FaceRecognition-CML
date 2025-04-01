import glob
import math
import os
import random
import shutil

import cv2
import json
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class YTFacesProcessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.faces_dir = os.path.join(self.dataset_dir, "Faces")
        self.face_index = {}
        self.largest_read_res = [0, 0]

    def process(self, invalidate_cache=False, frames_per_recording=-1):
        os.makedirs(self.faces_dir, exist_ok=True)
        npz_files = [os.path.join(root, file)
                     for root, _, files in os.walk(self.dataset_dir)
                     for file in files if file.endswith(".npz")]

        print(f"Found {len(npz_files)} recordings. Processing...")

        with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                refresh_per_second=2
        ) as progress:
            task = progress.add_task("Processing Files", total=len(npz_files))

            for file_path in npz_files:
                self.process_file(file_path, progress, invalidate_cache, frames_per_recording)
                progress.update(task, advance=1)

        with open(os.path.join(self.dataset_dir, "metadata.json"), "w") as f:
            json.dump({
                "largest_res": self.largest_read_res
            }, f, indent=4)

    def process_file(self, file_path, progress, invalidate_cache, frames_per_recording):
        try:
            with np.load(file_path, mmap_mode="r") as data:
                colour_images = data["colorImages"]
                name = os.path.basename(file_path).replace(".npz", "")
                name = "_".join(name.split("_")[:-1])
                name_dir = os.path.join(self.faces_dir, name)
                if os.path.exists(name_dir) and not invalidate_cache:
                    return

                os.makedirs(name_dir, exist_ok=True)

                num_frames = colour_images.shape[-1]
                if frames_per_recording > 0:
                    num_frames = min(num_frames, frames_per_recording)
                formatted_name = f"{name[:25]:<25}"

                frame_task = progress.add_task(f"[green]Frames {formatted_name}", total=num_frames)

                for i in range(num_frames):
                    frame = colour_images[..., i]
                    self.process_image(name_dir, frame, i + self.face_index.get(name, 0), invalidate_cache)
                    progress.update(frame_task, advance=1)

                self.face_index[name] = num_frames + self.face_index.get(name, 0)

                progress.remove_task(frame_task)

        except Exception as e:
            print(f"Error processing {file_path}", e)

    def process_image(self, name_dir, colour_image, index, invalidate_cache):
        face_path = os.path.join(name_dir, f"face_{index}.png")
        res = colour_image.shape[:2]
        if res[0] > self.largest_read_res[0]:
            self.largest_read_res[0] = res[0]
        if res[1] > self.largest_read_res[1]:
            self.largest_read_res[1] = res[1]
        if not os.path.exists(face_path) or invalidate_cache:
            rgb_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(face_path, rgb_image)


class YTFacesDataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.faces_dir = os.path.join(self.dataset_dir, "Faces")
        self.appropriate_res = None

    def load(self, frames_per_recording=-1, invalidate_split_cache=False, invalidate_faces_cache=False):
        if invalidate_split_cache:
            self.__invalidate_cache()
        if not os.path.exists(self.faces_dir) or invalidate_faces_cache:
            YTFacesProcessor(self.dataset_dir).process(invalidate_faces_cache, frames_per_recording)
        self.get_splits()

    def __invalidate_cache(self):
        self.appropriate_res = None

        def remove_split(split: str):
            split_dir = os.path.join(self.dataset_dir, split)
            if os.path.exists(split_dir):
                split_files = glob.glob(os.path.join(split_dir, "*")) + glob.glob(os.path.join(split_dir, ".*"))
                for file in split_files:
                    os.remove(file)
                os.rmdir(split_dir)

        remove_split("train")
        remove_split("test")

    def get_appropriate_res(self):
        if self.appropriate_res is not None:
            return self.appropriate_res

        self.appropriate_res = self.__get_largest_res()
        self.appropriate_res = [
            ((self.appropriate_res[0] + 15) // 16) * 16,
            ((self.appropriate_res[1] + 15) // 16) * 16
        ]
        return self.appropriate_res

    def __get_largest_res(self):
        if not os.path.exists(os.path.join(self.dataset_dir, "metadata.json")):
            raise Exception("Metadata file not found, run YTFacesProcessor first")
        with open(os.path.join(self.dataset_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            return metadata["largest_res"]

    def get_splits(self):
        splits = {}
        for split in ["train", "test"]:
            split_dir = os.path.join(self.dataset_dir, split)
            if not os.path.exists(split_dir):
                try:
                    self.__create_splits()
                except Exception as e:
                    print(f"Error creating splits, cleaning up: {e}")
                    import traceback
                    traceback.print_exc()
                    self.__invalidate_cache()
            splits[split] = split_dir
        return splits

    def __create_splits(self):
        for split in ["train", "test"]:
            split_dir = os.path.join(self.dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)

        people = os.listdir(self.faces_dir)
        random.shuffle(people)

        test_people = people[:int(0.2 * len(people))]
        train_people = people[int(0.2 * len(people)):]

        def move_faces(p, s):
            person_dir = os.path.join(self.faces_dir, p)
            split_dir = os.path.join(self.dataset_dir, s)

            if not os.path.isdir(person_dir):
                return

            for file in os.listdir(person_dir):
                if self.__pre_process(os.path.join(person_dir, file)):
                    shutil.copyfile(os.path.join(person_dir, file), os.path.join(split_dir, f"{p}_{file}"))


        print(f"Moving {len(test_people)} people to test and {len(train_people)} to train")

        for person in test_people:
            move_faces(person, "test")
        print("Test split done")

        for person in train_people:
            move_faces(person, "train")
        print("Train split done")

    def __pre_process(self, file):
        image = cv2.imread(file)
        res = image.shape[:2]
        appropriate_res = self.get_appropriate_res()
        left_pad = math.ceil((appropriate_res[0] - res[0]) / 2)
        right_pad = math.floor((appropriate_res[0] - res[0]) / 2)
        top_pad = math.ceil((appropriate_res[1] - res[1]) / 2)
        bottom_pad = math.floor((appropriate_res[1] - res[1]) / 2)
        try:
            image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cv2.imwrite(file, image)
            return True
        except Exception:
            return False

    