import os
import sys
from global_variables import *


def age_to_group(age):
    if age >= 18 and age<30:
        return 0
    elif age>55:
        return 1
    else:
        None


if __name__ == '__main__':

    # To read the data directory from the argument given
    user_path = sys.argv[1]

    txt = []

    for filename in os.listdir('/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized'):
        if '.DS_Store' in filename:
            continue
        print(filename)
        age = int(filename.split('_')[0])
        group = age_to_group(age)
        if group != None:
            strline = filename + ' %d\n' % group
            txt.append(strline)

    with open('files/gan_train.txt', 'w') as f:
        f.writelines(txt)
