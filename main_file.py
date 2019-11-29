# Standard library
import random
import sys

import PIL
# My library
sys.path.append('/content/Earth_Images_Dataset')
import network_class
# Third-party libraries
import numpy as np

from PIL import Image


import sys
import os
import csv

# Constants
LEARNING_RATES = [0.01]
gamma = 0.9
beta_1 = 0.9
beta_2 = 0.999
NUM_EPOCHS = 30
MINI_BATCH_SIZE = 20
act_func_is_sigmoid = 1 # 1 - sigmoid , 0 - tanh
cost_func_flag = 1 #1-Square Error , 2-cross entropy
plot_flag = 1
log_flag = 0
momentum_flag = 0  #0 - Standard GD , 1 - Momentum , 2 - nestorov, 3 - ADAM


def main():
    run_networks()

def get_train_images():

    myFileList = createFileList_train('/content/Earth_Images_Dataset/train_aug')
    basewidth = 320
    train_elements_generated = []
    k = 0
    for file in myFileList:
        k = k+1
        img_file = Image.open(file)

        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        # Resizing the Image
        wpercent = (basewidth / float(img_file.size[0]))
        hsize = int((float(img_file.size[1]) * float(wpercent)))
        img_file = img_file.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

        # Make image Greyscale
        img_grey = img_file.convert('L')
        # img_grey.save('result.png')
        if (k == 1):
            img_grey.show()

        # Save Greyscale values
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()

        train_elements_generated.append(list(value))
    print('Train Images Loaded')
    print(len(train_elements_generated))
    return (np.asarray(train_elements_generated))

def get_test_images():

    # CHANGE
    myFileList = createFileList_test('/content/Earth_Images_Dataset/test_aug')
    basewidth = 320
    test_elements_generated = []
    for file in myFileList:

        img_file = Image.open(file)

        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        # Resizing the Image
        wpercent = (basewidth / float(img_file.size[0]))
        hsize = int((float(img_file.size[1]) * float(wpercent)))
        img_file = img_file.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

        # Make image Greyscale
        img_grey = img_file.convert('L')
        # img_grey.save('result.png')
        # img_grey.show()

        # Save Greyscale values
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        test_elements_generated.append(list(value))

    print('Test Images Loaded')
    print(len(test_elements_generated))
    return (np.asarray(test_elements_generated))

def createFileList_train(myDir, format='.jpeg' ):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def createFileList_test(myDir, format='.jpg' ):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def load_data():

    # Get the testing Images
    test_data_ndarray = get_test_images()
    

    # Get the testing Labels
    with open('/content/Earth_Images_Dataset/new_test_labels_project.csv', newline='') as f2:
        l2 = list(csv.reader(f2))
    elements_2 = []
    for sub_list_2 in l2:
        for element in sub_list_2:
            temp = int(element)
            elements_2.append(temp)
    test_labels_ndarray = np.array(elements_2)
    print('Test Labels Loaded')
    test_data = (test_data_ndarray, test_labels_ndarray)

    # Get the training Images
    train_data_ndarray = get_train_images()

    # Get the training labels
    with open('/content/Earth_Images_Dataset/new_train_labels_project_aug.csv', newline='') as f2:
        l2 = list(csv.reader(f2))
    elements_2 = []
    for sub_list_2 in l2:
        for element in sub_list_2:
            temp = int(element)
            elements_2.append(temp)
    train_labels_ndarray = np.array(elements_2)
    print('Train Labels Loaded')
    training_data = (train_data_ndarray, train_labels_ndarray)

    return (training_data, test_data)

def load_data_wrapper():

    tr_d, te_d = load_data()

    training_inputs = [np.reshape(x, (58240, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x, (58240, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, test_data)



def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def run_networks():
    # Random.seed() is used so that every time weights and biases
    # are initialized with the same values
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, test_data = load_data_wrapper()

    for eta in LEARNING_RATES:
        #print("\nTrain a network using eta = "+str(eta))
        # Instantiate Network Object
        net = network_class.Network([58240, 50, 50, 10],
                                    act_func_is_sigmoid, cost_func_flag, plot_flag, log_flag
                                    , momentum_flag)
        # Run Stochastic Gradient Descent
        net.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE, eta, gamma, beta_1, beta_2, test_data)


if __name__ == "__main__":
    main()
