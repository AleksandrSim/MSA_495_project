from argparse import ArgumentParser

import mat73
import pandas as pd
import yaml

from global_variables import *
from features_from_data import *


class GenerateInitialFeatures:  # path of the original celebrity celebrity document. "NOTE" I read from local, cause it is too huge to uplaod to github
    def __init__(self, original_path=metadata_file):
        self.path = original_path
        self.data = mat73.loadmat(metadata_file)

    def generate_csv(self):
        name = [self.data['celebrityImageData']['name'][i][0] for i in
                range(len(self.data['celebrityImageData']['name']))]
        age = self.data['celebrityImageData']['age']
        dic = {'image_name': name, 'age': age}
        df = pd.DataFrame(dic)
        df.to_csv('imageName_age.csv')

    instance = GenerateInitialFeatures()
    instance.generate_csv()
