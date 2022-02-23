#from mtcnn.mtcnn import MTCNN
import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage import transform as trans
image_root_dir="/Users/aleksandrsimonyan/Desktop/CACD2000"
#image_root_dir=os.path.join(root_data_dir,"CACD2000")
#define store path
store_image_dir="/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized"
#store_image_dir=os.path.join(store_root_dir,"CACD2000")
if os.path.exists(store_image_dir) is False:
    os.makedirs(store_image_dir)

threshold = [0.6,0.7,0.9]
factor = 0.85
minSize=20
imgSize=[120, 100]
detector=MTCNN(steps_threshold=threshold,scale_factor=factor,min_face_size=minSize)

keypoint_list=['left_eye','right_eye','nose','mouth_left','mouth_right']



for filename in tqdm(os.listdir(image_root_dir)):
    dst = []
    filepath=os.path.join(image_root_dir,filename)
    storepath=os.path.join(store_image_dir,filename)
    npimage=np.array(Image.open(filepath))
    #Image.fromarray(npimage.astype(np.uint8)).show()

    dictface_list=detector.detect_faces(npimage)#if more than one face is detected, [0] means choose the first face

    if len(dictface_list)>1:
        boxs=[]
        for dictface in dictface_list:
            boxs.append(dictface['box'])
        center=np.array(npimage.shape[:2])/2
        boxs=np.array(boxs)
        face_center_y=boxs[:,0]+boxs[:,2]/2
        face_center_x=boxs[:,1]+boxs[:,3]/2
        face_center=np.column_stack((np.array(face_center_x),np.array(face_center_y)))
        distance=np.sqrt(np.sum(np.square(face_center - center),axis=1))
        min_id=np.argmin(distance)
        dictface=dictface_list[min_id]
    else:
        if len(dictface_list)==0:
            continue
        else:
            dictface=dictface_list[0]
    face_keypoint = dictface['keypoints']
    for keypoint in keypoint_list:
        dst.append(face_keypoint[keypoint])
    dst = np.array(dst).astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(npimage, M, (imgSize[1], imgSize[0]), borderValue=0.0)
    warped=cv2.resize(warped,(400,400))
    Image.fromarray(warped.astype(np.uint8)).save(storepath)
