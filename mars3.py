"""
    @author - Shaela Khan, 8th May, 2020 - Updated.
    Project - mars3.py the second version with the actual model code.
    Input - mars. dataset file.
    Output - object identified file with labels with supervised learned data.
            + labelled data with unidentified objects.

"""
from __future__ import print_function, division
# Ignore warnings
import warnings
import os
import time
import cv2
# importing the libraries
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# PyTorch libraries and modules
import torch
import torchvision
import torch.cuda
from torch import device
from torchvision import models
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models

# for reading and displaying images
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
print("Begin processing!.................")

# Process my ish!
IMAGES = np.array([])
instances = []
LABELS = []  # np.asarray([])


def process_my_data(myfile):
    # loading  dataset file.
    book = open(myfile, 'r')
    count = 0

    while True:  # testing
        count += 1

        # Get next line from file
        line = book.readline()
        img = line.split()
        img0 = "".join((img[:1]))  # now img0 is a string , instead of a list item

        lbl0 = "".join(img[1:])  # now extract the labels => should they be numeric?
        # print("extracted labels: content of lbl0: ", lbl0 + "   count:  ", count)

        # if line is empty
        # end of file is reached
        if not line:
            break
        else:
            # Extract image
            img1 = "./datasets/" + img0  # add the full path of image, missing from file,
            # print("debugging :", img1)
            # decided to go with, cv2.imread(), which saves as numpy matrix like we like.
            img2 = cv2.imread(img1)
            img2 = img2 / 255

            # resizing to fit in vgg16
            img2 = resize(img2, output_shape=(224, 224, 3), mode='constant', anti_aliasing=True)
            img2 = img2.astype('float32')
            # list of images.
            instances.append(img2)
            train_x = np.array(instances)
            # print("Shape of my image array    ", train_x.shape)
            # Now process the labels.
            # now lbl0 has integer form of the label.
            lbl0 = int(lbl0)
            lbl1 = np.binary_repr(lbl0)  # lbl1 has binary representation of labels
            print(" Printing label:bin   ", lbl0)

            # labels need to be updated for stuff. Do it later?
            LABELS.append(lbl0)
            train_y = np.array(LABELS)
    # print("Line{}: {}".format(count, line.strip()))
    book.close()
    return train_x, train_y


# Callers
train_data, train_labels = process_my_data('./datasets/train-short-500.txt')
print(" done processing train data.")
# test_data, test_labels = process_my_data('./datasets/test-shuffled-short-150.txt')
print("Shape of my image array training:   ", train_data.shape, " Shape of the labels array: ", train_labels.shape)

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.33,
                                                  random_state=42)  # removed the stratify option on here-
# print("\n Training dataset shape: ", train_x.shape, train_y.shape)
# print("\n Validation dataset shape: ", val_x.shape, val_y.shape)


# converting training images into torch format
train_x = train_x.reshape(335, 3, 224, 224)
train_x = torch.from_numpy(train_x)  # ??
# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# shape of training data
# train_x.shape, train_y.shape
print("\n Training dataset shape: ", train_x.shape, train_y.shape)

# converting validation images into torch format
val_x = val_x.reshape(165, 3, 224, 224)
val_x = torch.from_numpy(val_x)
# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)
# shape of validation data
print("\n Validation dataset shape: ", val_x.shape, val_y.shape)

# One final thing , class names -> known objects on Mars. Total = 25
class_names = {'apxs', 'apxs cal target', 'chemcam cal target',
               'chemin inlet open', 'drill', 'drill holes', 'drt front', 'drt side', 'ground',
               'horizon', 'inlet', 'mahli', 'mahli cal target',
               'mastcam', 'mastcam cal target', 'observation tray',
               'portion box', 'portion tube', 'portion tube opening', 'rems uv sensor', 'rover rear deck',
               'scoop', 'sun', 'turret', 'wheel'
               }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Part-2 - Data processing done.
# Modelling begins.
# -----------------------------------------------------------------------
# loading the pre-trained model
vgg_model = models.vgg16_bn(pretrained=True)
# Freeze model weights
for param in vgg_model.parameters():
    param.requires_grad = False

# checking if GPU is available
# if torch.cuda.is_available():
#   model = vgg_model.cuda()
# print(vgg_model)

# Modify the last layer.
number_features = vgg_model.classifier[6].in_features
features = list(vgg_model.classifier.children())[:-1]  # remove last layer
features.extend([torch.nn.Linear(number_features, len(class_names))])
vgg_model.classifier = torch.nn.Sequential(*features)

vgg_model = vgg_model.to(device)
print(vgg_model)

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)

# Train functions -from pytorch tutorial
# batch_size
batch_size = 24
# extracting features for train data.
data_x = []
label_x = []
inputs, labels = train_x, train_y

for i in tqdm(range(int(train_x.shape[0] / batch_size) + 1)):
    input_data = inputs[i * batch_size:(i + 1) * batch_size]
    label_data = labels[i * batch_size:(i + 1) * batch_size]
    input_data, label_data = Variable(input_data.cuda()), Variable(label_data.cuda())
    x = vgg_model.features(input_data)
    data_x.extend(x.data.cpu().numpy())
    label_x.extend(label_data.cpu().numpy())

# extracting features for validation data
data_y = []
label_y = []
inputs, labels = val_x, val_y

for i in tqdm(range(int(val_x.shape[0] / batch_size) + 1)):
    input_data = inputs[i * batch_size:(i + 1) * batch_size]
    label_data = labels[i * batch_size:(i + 1) * batch_size]
    input_data, label_data = Variable(input_data.cuda()), Variable(label_data.cuda())
    x = vgg_model.features(input_data)
    data_y.extend(x.data.cpu().numpy())
    label_y.extend(label_data.cpu().numpy())

# converting the features into torch format
x_train = torch.from_numpy(np.array(data_x))
x_train = x_train.view(x_train.size(0), -1)
y_train = torch.from_numpy(np.array(label_x))
x_val = torch.from_numpy(np.array(data_y))
x_val = x_val.view(x_val.size(0), -1)
y_val = torch.from_numpy(np.array(label_y))

# Train the model, working on it.
# batch size
# It’s time to train the model. We will train it for 30 epochs with a batch_size set to 128:
batch_size = 28

# number of epochs to train the model
n_epochs = 30

for epoch in tqdm(range(1, n_epochs + 1)):

    # keep track of training and validation loss
    train_loss = 0.0
    permutation = torch.randperm(x_train.size()[0])
    training_loss = []

    for i in range(0, x_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer_ft.zero_grad()
        # in case you wanted a semi-full example
        outputs = vgg_model.classifier(batch_x)
        loss = criterion(outputs, batch_y)
        training_loss.append(loss.item())
        loss.backward()
        optimizer_ft.step()

    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)

# prediction for training set
# ---------------------------------------------------------------------------------------------
prediction = []
target = []
permutation = torch.randperm(x_train.size()[0])

for i in tqdm(range(0, x_train.size()[0], batch_size)):
    indices = permutation[i:i + batch_size]
    batch_x, batch_y = x_train[indices], y_train[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = vgg_model.classifier(batch_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)

# training accuracy
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i], prediction[i]))

print('training accuracy: \t', np.average(accuracy))

# Last one to jump the ship.
print("End of transmission! ................")
