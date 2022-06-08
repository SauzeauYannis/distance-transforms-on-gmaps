import cv2

from distance_transform.preprocessing import reduce_image_size


def main():
    print("You can use this file to test your functions")
    image = cv2.imread("data/images/5_5_boundary.png", 0)
    image[2][4] = 0
    cv2.imwrite("data/images/5_5_boundary_2.png", image)

if __name__ == "__main__":
    main()
