import os
import sys
from global_variables import *


def age_to_group(age):
    if age <= 20:
        return 0
    elif 20 < age <= 30:
        return 1
    elif 30 < age <= 40:
        return 2
    elif 40 < age <= 50:
        return 3
    elif age > 50:
        return 4


if __name__ == '__main__':

    # To read the data directory from the argument given
    user_path = sys.argv[1]

    txt = []

    for filename in os.listdir(user_path):
        if '.DS_Store' in filename:
            continue
        print(filename)
        age = int(filename.split('_')[0])
        group = age_to_group(age)
        strline = filename + ' %d\n' % group

        txt.append(strline)

    with open('files/train.txt', 'w') as f:
        f.writelines(txt)
