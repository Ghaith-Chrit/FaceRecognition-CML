import os
import shutil
import argparse
import kagglehub
import config as cfg
import kagglehub.auth
import matplotlib.pyplot as plt

from mit_scenes import MITScenesProcessor
from yt_faces import YTFacesProcessor

def login():
    kagglehub.login()


def download_dataset_and_move(handle, path, force):
    if not os.path.exists(path) or force:
        orig_path = kagglehub.dataset_download(handle)
        print("Moving files...")
        shutil.move(orig_path, path)


def combine_face_and_face_easy_caltech(path):
    print("Combining easy and normal faces classes...")

    normal_faces_label = "Faces"
    easy_faces_label = "Faces_easy"

    normal_faces_path = os.path.join(path, normal_faces_label)
    easy_faces_path = os.path.join(path, easy_faces_label)

    if not os.path.isdir(easy_faces_path) or not os.listdir(easy_faces_path):
        return

    for filename in os.listdir(easy_faces_path):
        if not filename.startswith("fe_"):
            old_file_path = os.path.join(easy_faces_path, filename)
            new_file_path = os.path.join(easy_faces_path, f"fe_{filename}")
            os.rename(old_file_path, new_file_path)

    shutil.copytree(
        easy_faces_path,
        normal_faces_path,
        copy_function=shutil.move,
        dirs_exist_ok=True,
    )

    os.rmdir(easy_faces_path)


def combine_others_caltech(path):
    print("Combining all other classes...")

    faces_label = "Faces"
    other_label = "Other"

    other_folder = os.path.join(path, other_label)
    if not os.path.isdir(other_folder):
        os.makedirs(other_folder)

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if folder in [faces_label, other_label] or not os.path.isdir(folder_path):
            continue

        prefix = f"{folder[:2]}_"
        for filename in os.listdir(folder_path):
            if filename.startswith(prefix):
                continue

            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(other_folder, f"{prefix}{filename}")
            shutil.move(old_file_path, new_file_path)

        os.rmdir(folder_path)


def combine_all(src_path, dst_path):
    if not os.path.exists(src_path):
        return

    print("Combining all classes...")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for path, _, files in os.walk(src_path):
        if len(files) == 0:
            continue
        for file in files:
            dist_file = file
            if dist_file.startswith("img"):
                parts = os.path.normpath(path).split(os.sep)
                last_two_dirs = parts[-2:]
                class_name = class_name = "_".join(last_two_dirs)
                dist_file = f"{class_name}_{dist_file}"

            src_item_path = os.path.join(path, file)
            dst_item_path = os.path.join(dst_path, dist_file)
            shutil.move(src_item_path, dst_item_path)
        os.rmdir(path)
    os.rmdir(src_path)


def combine_non_person_classes_natural_images(src_path, dst_path):
    if not os.path.exists(src_path):
        return

    print("Combining all non-person classes...")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    images_path = os.path.join(src_path, "natural_images")

    for path, _, files in os.walk(images_path):
        if len(files) == 0:
            continue
        for file in files:
            dist_file = file
            if dist_file.startswith("person"):
                os.remove(os.path.join(path, file))
                continue
            src_item_path = os.path.join(path, file)
            dst_item_path = os.path.join(dst_path, dist_file)
            shutil.move(src_item_path, dst_item_path)
        os.rmdir(path)
    os.rmdir(images_path)


def print_report(face_dir, other_dir):
    faces_count = len(
        [f for f in os.listdir(face_dir) if os.path.isfile(os.path.join(face_dir, f))]
    )

    other_count = len(
        [f for f in os.listdir(other_dir) if os.path.isfile(os.path.join(other_dir, f))]
    )

    labels = ["Faces", "Other"]
    sizes = [faces_count, other_count]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Proportion of Faces vs Other Images in the Caltech Dataset")
    plt.show()


def process_caltech(root_dir):
    caltech_path = os.path.join(root_dir, cfg.DATA_CALTECH_101)
    caltech_101_path = os.path.join(caltech_path, "caltech-101")
    face_path = os.path.join(caltech_101_path, "Faces")
    other_path = os.path.join(caltech_101_path, "Other")

    download_dataset_and_move(cfg.CALTECH_101, caltech_path, args.force)
    combine_face_and_face_easy_caltech(caltech_101_path)
    combine_others_caltech(caltech_101_path)
    print_report(face_path, other_path)


def process_split(root_dir):
    face_dataset_path = os.path.join(root_dir, cfg.DATA_FACE_PATH)
    face_dist_path = os.path.join(face_dataset_path, "Face")
    face_dataset_actual_path = os.path.join(
        face_dataset_path, "lfw-deepfunneled", "lfw-deepfunneled"
    )

    download_dataset_and_move(cfg.FACE_DATASET, face_dataset_path, args.force)
    combine_all(face_dataset_actual_path, face_dist_path)

    other_dataset_path = os.path.join(root_dir, cfg.DATA_OTHER_PATH)
    other_dist_path = os.path.join(other_dataset_path, "Other")
    other_dataset_actual_path = os.path.join(other_dataset_path, "cifar10-128")

    download_dataset_and_move(cfg.OTHER_DATASET, other_dataset_path, args.force)
    combine_all(other_dataset_actual_path, other_dist_path)

def process_yt_faces(root_dir):
    yt_faces_path = os.path.join(root_dir, cfg.DATA_YT_FACES)

    download_dataset_and_move(cfg.YT_FACES, yt_faces_path, args.force)
    YTFacesProcessor(yt_faces_path).process()

def process_natural_images(root_dir):
    natural_images_path = os.path.join(root_dir, cfg.DATA_NATURAL_IMAGES)

    download_dataset_and_move(cfg.NATURAL_IMAGES, natural_images_path, args.force)
    combine_non_person_classes_natural_images(natural_images_path, os.path.join(natural_images_path, "Other"))

def process_scenes(root_dir):
    scenes_path = os.path.join(root_dir, cfg.DATA_SCENES)

    download_dataset_and_move(cfg.SCENES, scenes_path, args.force)
    MITScenesProcessor(scenes_path).process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        default="all",
        help="Options: all, caltech, split, yt_faces, natural_images, scenes, ade20k",
        choices=["all", "caltech", "split", "yt_faces", "natural_images", "scenes", "ade20k"],
    )

    parser.add_argument("--force", default=False, help="Redownload data")
    parser.add_argument(
        "--skip-login",
        default=False,
        help="Skip Kaggle Login via API (use if logging in interactively)",
    )

    args = parser.parse_args()
    ROOT_DIR = os.getcwd()

    if not args.skip_login:
        login()

    if args.dataset == "caltech" or args.dataset == "all":
        process_caltech(ROOT_DIR)

    if args.dataset == "split" or args.dataset == "all":
        process_split(ROOT_DIR)

    if args.dataset == "yt_faces" or args.dataset == "all":
        process_yt_faces(ROOT_DIR)

    if args.dataset == "natural_images" or args.dataset == "all":
        process_natural_images(ROOT_DIR)

    if args.dataset == "scenes" or args.dataset == "all":
        process_scenes(ROOT_DIR)
