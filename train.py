from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm
import numpy as np
import argparse
import imutils  # If you are unable to install this library, ask the TA; we only need this in extract_hsv_histogram.
import cv2
import os
import random
import skimage.io as io

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import time


# Depending on library versions on your system, one of the following imports 
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

path_to_dataset = r'datasetHOMUS_imgs_v3'
target_img_size = (32, 32) # fix image size because classification algorithms THAT WE WILL USE HERE expect that

# We are going to fix the random seed to make our experiments reproducible 
# since some algorithms use pseudorandom generators
random_seed = 42  
random.seed(random_seed)
np.random.seed(random_seed)

def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()

def extract_features(img, feature_set='hog'):
    ret = None
    
    ret = extract_hog_features(img) 
    return ret

def load_dataset(feature_set='hog'):
    features = []
    labels = []
    img_filenames = os.listdir(path_to_dataset)

    for i, fn in enumerate(img_filenames):
        if fn.split('.')[-1] != 'png':
        #if fn.split('.')[-1] != 'jpg':
            continue

        label = fn.split('_')[0]
        #label = fn.split('.')[0]
        labels.append(label)

        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path)
        #print("SHAPE: ",img.shape)
        features.append(extract_features(img, feature_set))
        
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_filenames)))
        
    return features, labels        



classifiers = {
    'NN': MLPClassifier(solver='sgd', random_state=random_seed, hidden_layer_sizes=(400,400), max_iter=500, verbose=1)
}

# This function will test all our classifiers on a specific feature set
def run_experiment(feature_set):
    
    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(feature_set)
    print('Finished loading dataset.')
    
    # Since we don't want to know the performance of our classifier on images it has seen before
    # we are going to withhold some images that we will test the classifier on after training 
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)
    
    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)
        
        # Test the model on images it hasn't seen before
        accuracy = model.score(test_features, test_labels)
        
        print(model_name, 'accuracy:', accuracy*100, '%')
        
        
t = time.time()
run_experiment('hog')

print("Training time in minutes: ", (time.time()-t)/60)

nn = classifiers['NN']
filename = 'neural_model.sav'
pickle.dump(nn, open(filename, 'wb'))