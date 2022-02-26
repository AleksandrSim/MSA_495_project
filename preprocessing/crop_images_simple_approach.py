import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from global_variables import *
from helper_functions import *


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def shape_to_normal(shape):
    shape_normal = []
    for index in range(0, 5):
        shape_normal.append((index, (shape.part(index).x, shape.part(index).y)))
    return shape_normal


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


if __name__ == '__main__':

    print("Image Cropping Starting..")

    path_main = os.path.split(os.getcwd())[0]

    generate_dir_if_not_exists(clean_images_output_path)
    generate_dir_if_not_exists(excluded_images_output_path)

    df = pd.read_csv(path_main + '/files/cleaned_images.csv')

    df = df[['name', 'class']]
    counter = 0
    exclusion_counter = 0
    for i in range(len(df)):
        print(f"\rImages processed -> {counter} and blurry images removed -> {exclusion_counter}", end='')
        sys.stdout.flush()
        counter += 1
        image = cv2.imread(original_images_path + "/" + df['name'][i])

        try:
            # Converting the image into grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(image)
            # if the focus measure is less than the supplied threshold,
            # then the image should be considered "blurry"
            if fm < images_blur_threshold:
                exclusion_counter += 1
                cv2.imwrite(excluded_images_output_path + "/" + df['name'][i], image)
                continue

            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

            # Creating variable faces
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                exclusion_counter += 1
                cv2.imwrite(excluded_images_output_path + "/" + df['name'][i], image)
                continue

            for (x, y, w, h) in faces:
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Creating two regions of interest
                roi_gray = gray[y:(y + h), x:(x + w)]
                roi_color = image[y:(y + h), x:(x + w)]

            # Creating variable eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

            if len(eyes) == 0:
                exclusion_counter += 1
                cv2.imwrite(excluded_images_output_path + "/" + df['name'][i], image)
                continue


            cv2.imwrite(clean_images_output_path + "/" + df['name'][i], image)

            '''
                
        
            index = 0
            # Creating for loop in order to divide one eye from another
            for (ex, ey, ew, eh) in eyes:
                if index == 0:
                    eye_1 = (ex, ey, ew, eh)
                elif index == 1:
                    eye_2 = (ex, ey, ew, eh)
                # Drawing rectangles around the eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
                index = index + 1

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # Calculating coordinates of a central points of the rectangles
            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]

            right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]

            cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
            cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
            cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)

            if left_eye_y > right_eye_y:
                A = (right_eye_x, left_eye_y)
                # Integer -1 indicates that the image will rotate in the clockwise direction
                direction = -1
            else:
                A = (left_eye_x, right_eye_y)
                # Integer 1 indicates that image will rotate in the counter clockwise
                # direction
                direction = 1

            cv2.circle(roi_color, A, 5, (255, 0, 0), -1)

            cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)
            cv2.line(roi_color, left_eye_center, A, (0, 200, 200), 3)
            cv2.line(roi_color, right_eye_center, A, (0, 200, 200), 3)

            delta_x = right_eye_x - left_eye_x
            delta_y = right_eye_y - left_eye_y
            angle = np.arctan(delta_y / delta_x)
            angle = (angle * 180) / np.pi

            # Width and height of the image
            #h, w = image.shape[:2]
            # Calculating a center point of the image
            # Integer division "//"" ensures that we receive whole numbers
            #center = (w // 2, h // 2)
            # Defining a matrix M and calling
            # cv2.getRotationMatrix2D method
            #M = cv2.getRotationMatrix2D(center, (angle), 1.0)
            # Applying the rotation to our image using the
            # cv2.warpAffine method
            #rotated = cv2.warpAffine(image, M, (w, h))
            
            '''

        except:
            exclusion_counter += 1
            cv2.imwrite(excluded_images_output_path + "/" + df['name'][i], image)


'''
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faces = image[y:y + h, x:x + w]
            warped = cv2.resize(faces, (400, 400))
            cv2.imwrite(clean_images_output_path + "/" + df['name'][i], faces)

        Image.fromarray(faces.astype(np.uint8)).save(clean_images_output_path + "/" + df['name'][i])
'''
