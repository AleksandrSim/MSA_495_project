import argparse
import pandas as pd
import matplotlib.pyplot as plt
from main import *
import seaborn as sns
sns.set()


def argumentParser():
    parser = argparse.ArgumentParser(
        description= 'path to the the dataset_of_images'
    )
    parser.add_argument('-i', '--input', required= True, help= 'path_to_the_dataset')
    args = parser.parse_args()
    return args

dataset = argumentParser()
path = dataset.input



def group_age_decades(path='imageName_age.csv'):
    df = pd.read_csv(path)
    lis_age = list(df['age'])
    decades = [(int(str(i)[0]) *10+5) for i in lis_age]
    df['decades'] = decades
    return df


def generate_of_age_distribution(df):
    df = group_age_decades(path='imageName_age.csv')
    data_by_age_decade = df.groupby(['decades']).count().reset_index() # We will need this later
    decades = df['decades']
    counts = df['age']
    sns.countplot(x = decades)
    plt.show()


df =  group_age_decades()
df['old_young'] =  [1 if int(i) > 50 else 0 for i in list(df['age']) ]
sn
old_young = list(df['old_young'])

sns.countplot(x= df['old_young'])  #  LOST YOU Are you here ?? :D :D # We need to delete the comments later or we will be fuckedu up
plt.show()

generate_of_age_distribution(df)