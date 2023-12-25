import cv2
import numpy as np
import os

def read_image(root_path="./images", image_name="1.jpg"):
    path = os.path.join(root_path, image_name)
    image = cv2.imread(path)
    return image


if __name__ == "__main__":
    print(read_image().shape)
    