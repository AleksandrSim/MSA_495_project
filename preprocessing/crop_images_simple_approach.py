import sys
import math
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from global_variables import *
from helper_functions import *


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def detectFace(img):

    if img is None:
        return None, None

    faces = face_detector.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=3)

    if len(faces) > 0:
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        img = img[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray
    else:
        return None, None


def alignFace(img):
    # plt.imshow(img[:, :, ::-1])
    # plt.show()

    img_raw = img.copy()

    img, gray_img = detectFace(img)

    eyes = eye_detector.detectMultiScale(gray_img)

    if len(eyes) >= 2:
        # find the largest 2 eye
        base_eyes = eyes[:, 2]
        # print(base_eyes)
        items = []
        for index, eye in enumerate(base_eyes):
            item = (eye, index)
            items.append(item)
        df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(by=['length'], ascending=False)
        eyes = eyes[df.idx.values[0:2]]
        # --------------------
        # decide left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        # --------------------
        # center of eyes
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 2)

        # ----------------------
        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
            # print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            # print("rotate to inverse clock direction")

        # ----------------------

        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)

        cv2.line(img, right_eye_center, left_eye_center, (67, 67, 67), 1)
        cv2.line(img, left_eye_center, point_3rd, (67, 67, 67), 1)
        cv2.line(img, right_eye_center, point_3rd, (67, 67, 67), 1)

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)

        angle = (angle * 180) / math.pi

        if direction == -1:
            angle = 90 - angle

        # --------------------
        # rotate image

        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))

        return new_img

    return img_raw


if __name__ == '__main__':

    print("Image Cropping Starting..")

    path_main = os.path.split(os.getcwd())[0]

    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    face_detector_path = path + "/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path + "/data/haarcascade_eye.xml"
    nose_detector_path = path + "/data/haarcascade_mcs_nose.xml"

    generate_dir_if_not_exists(clean_images_output_path)
    generate_dir_if_not_exists(blurry_images_path)
    generate_dir_if_not_exists(excluded_images_path)

    df = pd.read_csv(path_main + '/files/cleaned_images.csv')

    df = df[['name', 'class']]
    counter = 0
    exclusion_counter = 0
    for i in range(len(df)):

        if file_exists(clean_images_output_path + df['name'][i]) or file_exists(
                blurry_images_path + df['name'][i]) or file_exists(excluded_images_path + df['name'][i]):
            continue

        image = cv2.imread(original_images_path + df['name'][i])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fm = variance_of_laplacian(image)
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm < images_blur_threshold:
            exclusion_counter += 1
            Image.fromarray(image.astype(np.uint8)).save(blurry_images_path + df['name'][i])
            continue

        face_detector = cv2.CascadeClassifier(face_detector_path)
        eye_detector = cv2.CascadeClassifier(eye_detector_path)
        nose_detector = cv2.CascadeClassifier(nose_detector_path)

        alignedFace = alignFace(image)
        # plt.imshow(alignedFace[:, :, ::-1])
        # plt.show()
        img, gray_img = detectFace(alignFace)
        # plt.imshow(img[:, :, ::-1])
        # plt.show()

        if img is not None:
            img = cv2.resize(img, image_dimensions)
            #cv2.imwrite(clean_images_output_path + df['name'][i], img)
            Image.fromarray(img.astype(np.uint8)).save(clean_images_output_path + df['name'][i])
        else:
            exclusion_counter += 1
            Image.fromarray(image.astype(np.uint8)).save(excluded_images_path + df['name'][i])
            # cv2.imwrite(excluded_images_path + df['name'][i], image)

            continue

        counter += 1
        print(f"\rImages processed -> {counter} and number of excluded images -> {exclusion_counter}", end='')
        sys.stdout.flush()
