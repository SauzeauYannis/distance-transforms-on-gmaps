"""
Utils
"""
from pathlib import Path


def find_images_with_stomata(dir_path: str, out_dir_path: str) -> None:
    num_images = 0
    num_labeled_images = 0
    num_cross = 0
    num_long = 0
    num_paradermal = 0
    for path in Path(dir_path).rglob('*.png'):
        print(path.name)
        num_images += 1
        if "Label" in path.name:
            num_labeled_images += 1
            if "cros" in path.name:
                num_cross += 1
            if "long" in path.name:
                num_long += 1
            if "para" in path.name:
                num_paradermal += 1

    print(num_images)
    print(num_labeled_images)
    print(num_paradermal)
    print(num_cross)
    print(num_long)


def main():
    find_images_with_stomata("C:/Documenti/UNISA/II_Magistrale/erasmus/thesis_work/data/labeled_slices",
                             "C:/Documenti/UNISA/II_Magistrale/erasmus/thesis_work/data/labels_with_stomata")


if __name__ == "__main__":
    main()
