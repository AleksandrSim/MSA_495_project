
path_to_dataset_resized = '/Users/aleksandrsimonyan/Desktop/cross_age_dataset_cleaned_and_resized'
import os
def age_to_group(age):
    if age <= 20:
        return 0
    elif age > 20 and age <= 30:
        return 1
    elif age > 30 and age <= 40:
        return 2
    elif age > 40 and age <= 50:
        return 3
    elif age > 50:
        return 4
if __name__ =='__main__':



    txt = []

    for filename in os.listdir(path_to_dataset_resized):
        if '.DS_Store' in filename:
            continue
        print(filename)
        age = int(filename.split('_')[0])
        group = age_to_group(age)
        strline = filename + ' %d\n' % group

        txt.append(strline)

    with open('files/train.txt', 'w') as f:
        f.writelines(txt)

