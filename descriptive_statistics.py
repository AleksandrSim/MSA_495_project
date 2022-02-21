import argparse
import pandas as pd
import matplotlib.pyplot as plt
from main import *
import seaborn as sns
sns.set()
sns.set(rc={'figure.figsize':(29.7/2,20.27/2)})

sns.set(font_scale = 1.9)

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



def generate_class_and_plot(df):
    df =  group_age_decades()
    df['old_young'] =  [1 if int(i) > 50 else 0 for i in list(df['age']) ]
    old_young = list(df['old_young'])
    sns.countplot(x= df['old_young'])
    plt.show()
    return df


def break_names(df):
    df = group_age_decades()
    df = generate_class_and_plot(df)
    df['name'] = df['image_name'].str.split('_').str[1] +'_' + df['image_name'].str.split('_').str[2]
    return df




def most_occuring_actors(df):
    name_count = df.groupby(['name']).count().reset_index()

    name_count = name_count[['name', 'age']]
    name_count.columns = ['name', 'count']

    sns.barplot(x='name', y='count', data=name_count[name_count['count'] > 133])
    plt.show()
    print('-------->mean' + " "+ str(name_count['count'].mean()))
    print('-------->median'+ " "+ str(name_count['count'].median()))

if __name__ =='__main__':
    df = group_age_decades()
    df =  generate_of_age_distribution(df)
    df = generate_class_and_plot(df)
    df =  break_names(df)
    df =  most_occuring_actors(df)

