import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



if __name__ =='__main__':
    path_main =os.path.split(os.getcwd())[0]

    df = pd.read_csv(path_main + '/files/cleaned_images.csv')
    path_to_images = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/'  # make this dinamic and change the path

    df = df[['name','class']]
    for i in range(len(df)):
        image = cv2.imread('/Users/aleksandrsimonyan/Desktop/CACD2000/' + df['name'][i])
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
        Image.fromarray(warped.astype(np.uint8)).save(
            '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized/' + df['name'][i])







