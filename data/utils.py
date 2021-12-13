"""
Utils
"""
from pathlib import Path
from shutil import copyfile
import os
import cv2
import labels
from distance_transform.preprocessing import *


def contains_stomata(image_path) -> bool:
    image = cv2.imread(str(image_path), 0)
    print(get_different_values_image(image))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == labels.labels["stomata"]:
                print(f"i,j: {(i,j)}")
                print("True")
                return True

    return False


def find_images_with_stomata(dir_path: str, out_dir_path: str) -> None:
    num_images = 0
    num_labeled_images = 0
    num_cross = 0
    num_cross_with_stomata = 0
    num_long_with_stomata = 0
    num_long = 0
    num_paradermal = 0
    cross_folder = out_dir_path + "cross/"
    long_folder = out_dir_path + "long/"
    for path in Path(dir_path).rglob('*.png'):
        print(path.name)
        num_images += 1
        if "Label" in path.name:
            num_labeled_images += 1
            if "cros" in path.name:
                num_cross += 1
                if contains_stomata(path):
                    num_cross_with_stomata += 1
                    copyfile(path, cross_folder + path.name)
            if "long" in path.name:
                num_long += 1
                if contains_stomata(path):
                    num_long_with_stomata += 1
                    copyfile(path, long_folder + path.name)
            if "para" in path.name:
                num_paradermal += 1

    print(num_images)
    print(num_labeled_images)
    print(num_paradermal)
    print(num_cross)
    print(num_long)
    print(f"num_cross_with_stomata: {num_cross_with_stomata}")
    print(f"num_long_with_stomata: {num_long_with_stomata}")


def main():
    find_images_with_stomata("C:/Documenti/UNISA/II_Magistrale/erasmus/thesis_work/data/labeled_slices",
                             "labels_with_stomata/")


if __name__ == "__main__":
    main()
