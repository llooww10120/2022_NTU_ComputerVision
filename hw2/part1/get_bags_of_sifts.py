from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    pkl = open('vocab.pkl', 'rb')
    vocab = pickle.load(pkl)
    step = 10
    image_feats = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        img = np.array(img)

        frames, descriptors = dsift(img, step=[step,step], window_size=4, fast=True)
        descriptors =descriptors[::5]
        dis = distance.cdist(vocab, descriptors)  
        k = np.argmin(dis, axis = 0)
        hist, bin_edges = np.histogram(k, bins=len(vocab))
        sum = k.shape[0]
        norm = hist/sum
        image_feats.append(norm)
    image_feats = np.array(image_feats)
    
    # print('bags end')
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
