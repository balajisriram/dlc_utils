
from tqdm import tqdm
import collections
import sys
import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import time
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import seaborn as sns


# Function to determine number of clusters for KMeans and GMM
# input: method:'GMM' or 'KMeans'; data: dataframe;range_a,range_b: cluster k range

def determineK(method,data,range_a,range_b):
    if (method == 'GMM'):
        
        start_time = time.time()
        n_components = np.arange(range_a, range_b)
        models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data) for n in tqdm(n_components)]

        print("---GMM determine k: %s seconds ---" % (time.time() - start_time))
        
        plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
        plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('No of clusters')
        plt.show()
        
    elif (method == 'KMeans'):
        
        start_time = time.time()
        Error =[]
        for i in tqdm(range(range_a, range_b)):
            kmeans = KMeans(n_clusters = i).fit(data)
            kmeans.fit(data)
            Error.append(kmeans.inertia_)
            print(i)
        print("---Elbow method: %s seconds ---" % (time.time() - start_time))

        plt.plot(range(range_a, range_b), Error)
        plt.title('Elbow method')
        plt.xlabel('No of clusters')
        plt.ylabel('Error')
        plt.show()



# Dimension reduction using PCA
# return a dataframe containing PCs
# input: x = dataset; 
# input: sd = 1 perform w/t standardization, sd = 0 perform w/o standardization
# output 1: var = cumulated variance explained
# output 2: pca = Princepal components 
def behavior_pca(x,sd):


    # PCA with or without standardization
    pca = PCA()
    if sd == 0:
        x_transformed = pca.fit_transform(x)
    else:
        x_sd = StandardScaler().fit_transform(x)
        x_transformed = pca.fit_transform(x_sd)

    PCA_components = pd.DataFrame(x_transformed)
    
    features = range(pca.n_components_)

    fig = plt.figure(figsize=(15, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    #fig.suptitle('PCA Analysis')

    # Plot the explained variances by each PC
    ax1.bar(features, pca.explained_variance_ratio_, color='blue')
    ax1.title.set_text('Variance Explained by Single PC')

    # Plot the cumulative explained variances
    # Cumulative sum of variance explained with [n] features
    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    var = np.around(var, decimals=2)
    ax2.bar(features, var, color='blue')
    ax2.title.set_text('cumulative Variance Explained')

    return var,PCA_components



# gmm clustering
# input: tr_d = training data, te_d = test data, k = no of clusters, dim = 2 or 3 dims of cluster plot
# output: an array of cluster labels
def gmm_plt(tr_d,te_d,k,dim):
    gmm = GaussianMixture(n_components=k)
    gmm.fit(tr_d)
    labels = gmm.predict(te_d)
    if (dim==2):
        plt.scatter(te_d[:, 0], te_d[:, 1], c=labels, s=1,alpha=0.1,cmap='rainbow')
        plt.title('GMM Clustering in 2 Dimensions')
    elif (dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(te_d[:, 0], te_d[:, 1], te_d[:, 2], c=labels,s=1,alpha=0.1, cmap='rainbow')
        plt.title('GMM Clustering with 3 Dimensions')
    return labels 

# TSNE dimension reduction
# input: data = input data
# output: tsne_results = 2 dimensional TSNE data
def behavior_tsne(data):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return tsne_results



# Draw 10 skeleton plots for each cluster
# input: data = data; c = cluter no.
def cluster_visual(data,c):
    data_c = data.loc[(data['Pred']==c)]
    sample_data_c = data_c.sample(10)
    gs = plt.GridSpec(5, 2)
    plt.figure(figsize=(5,20))
  
    plt.title("Cluster "+ str(c))
    for i in range(10):
        a = i%2
        b = i//2

        plt.subplot(gs[b,a])
        plt.xlim(xmax=15,xmin=-15)
        plt.ylim(ymax=50,ymin=-50)
        plt.scatter(-12,-45,s=20*sample_data_c.iloc[i,36], label='velocity', lw=1)
        plt.plot([0, sample_data_c.iloc[i,6]], [0,sample_data_c.iloc[i,7]],label = "snout")
        plt.plot([0, sample_data_c.iloc[i,8]], [0,sample_data_c.iloc[i,9]],label = "tailbase")
        plt.plot([0, sample_data_c.iloc[i,10]], [0,sample_data_c.iloc[i,11]],label = "left-ear")
        plt.plot([0, sample_data_c.iloc[i,12]], [0,sample_data_c.iloc[i,13]],label = "right-ear")

        plt.plot([sample_data_c.iloc[i,10], sample_data_c.iloc[i,6]], [sample_data_c.iloc[i,11],sample_data_c.iloc[i,7]],
                 dashes=[10, 5, 10, 5],color = "black")
        plt.plot([sample_data_c.iloc[i,12], sample_data_c.iloc[i,6]], [sample_data_c.iloc[i,13],sample_data_c.iloc[i,7]],
                 dashes=[10, 5, 10, 5],color = "black")
        plt.scatter(sample_data_c.iloc[i,6],sample_data_c.iloc[i,7],s=20,color = "blue")
    plt.legend(loc='upper right',fontsize=7)
    
    plt.show()
        