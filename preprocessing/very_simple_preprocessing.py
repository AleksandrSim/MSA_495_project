from argparse import ArgumentParser

import yaml
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
from global_variables import *
from helper_functions import *
import sys
import os

if __name__ == '__main__':

    print("Image Cropping Starting..")

    parser = ArgumentParser()
    parser.add_argument('--config', default='../config_files/config.yaml', help='Config .yaml file to use for training')

    # To read the data directory from the argument given
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)


    # To read the data directory from the argument given
    user_path = config['user_path']

    generate_dir_if_not_exists(user_path + clean_images_path)
    generate_dir_if_not_exists(user_path + excluded_images_path)

    df = pd.read_csv(os.path.split(os.getcwd())[0] + '/files/cleaned_images.csv')

    df = df[['name', 'class']]
    counter = exclusion_counter = 0
    for i in range(len(df)):

        if file_exists(user_path + clean_images_path + df['name'][i]) or file_exists(user_path + excluded_images_path + df['name'][i]):
            continue

        image = cv2.imread(user_path + original_images_path + df['name'][i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30))
        if len(faces) == 0:
            exclusion_counter += 1
            resized = cv2.resize(image, image_dimensions)
            Image.fromarray(image.astype(np.uint8)).save(user_path + excluded_images_path + df['name'][i])
            continue
        for (x, y, w, h) in faces:
            faces = image[y:y + h, x:x + w]
            warped = cv2.resize(faces, image_dimensions)
        Image.fromarray(warped.astype(np.uint8)).save(user_path + clean_images_path + df['name'][i])
        counter += 1
        print(f"\rImages processed -> {counter} and number of excluded images -> {exclusion_counter}", end='')
        sys.stdout.flush()
