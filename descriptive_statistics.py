import argparse
import pandas as pd
import matplotlib.pyplot as plt
from main import *


def argumentParser():
    parser = argparse.ArgumentParser(
        description= 'path to the the dataset_of_images'
    )
    parser.add_argument('-i', '--input', required= True, help= 'path_to_the_dataset')
    args = parser.parse_args()
    return args

dataset = argumentParser()
path = dataset.input


pd.read_csv('imageName_age.csv')