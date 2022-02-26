from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
from global_variables import *
from helper_functions import *

if __name__ == '__main__':

    print("Image Cropping Starting..")

    path_main = os.path.split(os.getcwd())[0]

    generate_dir_if_not_exists(clean_images_output_path)

    df = pd.read_csv(path_main + '/files/cleaned_images.csv')

    df = df[['name', 'class']]
    counter = 0
    for i in range(len(df)):
        if counter % 100 == 0:
            print("Images Processed: " + str(counter))
        image = cv2.imread(original_images_path + "/" + df['name'][i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30))
        if len(faces) == 0:
            continue
        for (x, y, w, h) in faces:
            faces = image[y:y + h, x:x + w]
            warped = cv2.resize(faces, (400, 400))
        Image.fromarray(warped.astype(np.uint8)).save(clean_images_output_path + "/" + df['name'][i])
        counter += 1

