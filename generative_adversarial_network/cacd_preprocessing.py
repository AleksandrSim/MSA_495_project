import os
import shutil
from argparse import ArgumentParser
import yaml
from scipy.io import loadmat
from global_variables import *
from helper_functions import *
import pandas as pd
import numpy as np


def copy_images_to_path(domain, image, user_path):
    if file_exists(os.path.join(user_path + clean_images_path, image)) and not file_exists(os.path.join(domain, image)):
        shutil.copy(os.path.join(user_path + clean_images_path, image), os.path.join(domain, image))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='../config_files/config.yaml', help='Config .yaml file to use for training')

    # To read the data directory from the argument given
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    # To read the data directory from the argument given
    user_path = config['user_path']

    metadata = loadmat(metadata_file)['celebrityImageData'][0][0]
    ages = [x[0] for x in metadata[0]]
    names = [x[0][0] for x in metadata[-1]]

    '''
    # converting list to array
    df_describe = pd.DataFrame(np.array(ages))
    df_describe.describe()
    print("Mean is --> " + str(np.mean(np.array(ages), axis=0)))
    print("Median is --> " + str(np.median(np.array(ages), axis=0)))
    print("Max is --> " + str(np.max(np.array(ages), axis=0)))
    print("Min is --> " + str(np.min(np.array(ages), axis=0)))
    '''

    numpy_ages_array = np.array(ages)

    min_age = np.min(numpy_ages_array)
    max_age = np.min(numpy_ages_array)

    ages_to_keep_a = [x for x in range(min_age, 20)]
    ages_to_keep_b = [x for x in range(20, 30)]
    ages_to_keep_c = [x for x in range(30, 40)]
    ages_to_keep_d = [x for x in range(40, 50)]
    ages_to_keep_e = [x for x in range(50, max_age + 1)]

    domainA = domainB = domainC = domainD = domainE = []

    for age, name in zip(ages, names):
        if age in ages_to_keep_a:
            domainA.append(name)
        if age in ages_to_keep_b:
            domainB.append(name)
        if age in ages_to_keep_c:
            domainC.append(name)
        if age in ages_to_keep_d:
            domainD.append(name)
        if age in ages_to_keep_e:
            domainE.append(name)

    N = min(len(domainA), len(domainB), len(domainC), len(domainD), len(domainE))
    domainA = domainA[:N]
    domainB = domainB[:N]
    print(f'Images in A -> {len(domainA)}, -> B {len(domainB)}, -> C {len(domainC)}, -> D {len(domainD)}, -> E {len(domainE)}')

    domainA_dir = os.path.join(user_path + gan_input_images, 'trainA')
    domainB_dir = os.path.join(user_path + gan_input_images, 'trainB')
    domainC_dir = os.path.join(user_path + gan_input_images, 'trainC')
    domainD_dir = os.path.join(user_path + gan_input_images, 'trainD')
    domainE_dir = os.path.join(user_path + gan_input_images, 'trainE')

    generate_dir_if_not_exists(domainA_dir)
    generate_dir_if_not_exists(domainB_dir)
    generate_dir_if_not_exists(domainC_dir)
    generate_dir_if_not_exists(domainD_dir)
    generate_dir_if_not_exists(domainE_dir)

    for imageA, imageB, imageC, imageD, imageE in zip(domainA, domainB, domainC, domainD, domainE):
        copy_images_to_path(domainA_dir, imageA, user_path)
        copy_images_to_path(domainB_dir, imageB, user_path)
        copy_images_to_path(domainC_dir, imageC, user_path)
        copy_images_to_path(domainD_dir, imageD, user_path)
        copy_images_to_path(domainE_dir, imageE, user_path)
