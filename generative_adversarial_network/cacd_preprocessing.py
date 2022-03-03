import os
import shutil
from argparse import ArgumentParser
import yaml
from scipy.io import loadmat
from global_variables import *
from helper_functions import *


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

    ages_to_keep_a = [x for x in range(18, 30)]
    ages_to_keep_b = [x for x in range(55, 100)]

    domainA, domainB = [], []
    for age, name in zip(ages, names):
        if age in ages_to_keep_a:
            domainA.append(name)
        if age in ages_to_keep_b:
            domainB.append(name)

    N = min(len(domainA), len(domainB))
    domainA = domainA[:N]
    domainB = domainB[:N]
    print(f'Images in A {len(domainA)} and B {len(domainB)}')

    domainA_dir = os.path.join(user_path + gan_input_images, 'trainA')
    domainB_dir = os.path.join(user_path + gan_input_images, 'trainB')

    generate_dir_if_not_exists(domainA_dir)
    generate_dir_if_not_exists(domainB_dir)

    for imageA, imageB in zip(domainA, domainB):
        if file_exists(os.path.join(user_path + clean_images_path, imageA)) and not file_exists(os.path.join(domainA_dir, imageA)):
            shutil.copy(os.path.join(user_path + clean_images_path, imageA), os.path.join(domainA_dir, imageA))
        if file_exists(os.path.join(user_path + clean_images_path, imageB)) and not file_exists(os.path.join(domainB_dir, imageB)):
            shutil.copy(os.path.join(user_path + clean_images_path, imageB), os.path.join(domainB_dir, imageB))
