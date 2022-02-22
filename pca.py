import mat73
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(path):
    data_dict = mat73.loadmat(path)

    return data_dict

def pca_plots(data_dict):
    features = data_dict['celebrityImageData']['feature']
    df = pd.DataFrame(features)

    pca = PCA()
    pca.fit(df)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    fig = plt.figure(figsize=(8, 8))
    plt.plot(cumsum)
    plt.xlabel('PCA features')
    plt.ylabel('Cumulative Variance Explained')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('Variance Explained')
    plt.show()

    pca = PCA(n_components=500)
    fit = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca.components_)

    return None

if  __name__=='__main__':
    path = 'project_data/celebrity2000.mat'
    data_dict = load_data(path)
