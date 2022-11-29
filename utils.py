import glob
import random
import os
import cv2
import csv
from datetime import datetime

import config


def rand_image(images_dir):
    """Returns random image from input image directory.

    Args:
        images_dir: Path directory of images folder.
    Returns:
        image_name: image name of the associated image path
        image_path: image path of the randomly selected image
    """
    image_paths = glob.glob(f"{images_dir}/*.jpg")

    image_path = random.choice(image_paths)
    image_name = os.path.basename(os.path.normpath(image_path))
    return image_name, image_path

def check_dir_integrity():
    """Verifies folder dependencies exist and creates them if not.
    """
    folder_list = ['exports', 'logs']
    for folder_name in folder_list:
        path = f'{os.getcwd()}/{folder_name}'
        
        is_exist = os.path.exists(path)
        if not is_exist:
            os.makedirs(path)

def export_image(image):
    """Writes image data of the input np.array image to disk.

    Args:
        image: input np.array of the image
    Returns:
        export_path: path of the exported image
    """
    path = f'{os.getcwd()}\exports'
    file_count = 0 if len(os.listdir(path)) <= 0 else (len(os.listdir(path)))
    
    if file_count >= 9999:
        raise Exception("Error: Export folder is full.")

    image_name = f'Image{str(file_count).zfill(4)}.jpg'
    export_path = f'{os.getcwd()}\exports\{image_name}'

    resized_image = cv2.resize(image, (config.EXPORT_IMAGE_WIDTH, config.EXPORT_IMAGE_HEIGHT))
    cv2.imwrite(export_path, resized_image)
    return export_path

def logging(file_path, det_conf):
    """Logs the detection data onto the associated logging.csv file

    Args:
        file_path: file path of the exported drawn-on image in reference to the detection
        det_conf: confidence score value of the detection output
    """
    csv_path = f'{os.getcwd()}\logs\logging.csv'
    if not os.path.isfile(csv_path):
        header = ['date', 'time', 'file_path', 'label', 'conf_score']
        with open(csv_path, "w", newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(header)

    curr_timestamp = str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")).split()
    new_line = [curr_timestamp[0], curr_timestamp[1], file_path, 'package', f'{det_conf}%']
    with open(csv_path, "a", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(new_line)
