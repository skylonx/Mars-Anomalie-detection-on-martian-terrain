"""
    @author - Shaela Khan, 11th May, 2020 - Updated.
    Project - mars on ice.py the second version with the actual model code.
    Input - mars. dataset file.
    Output - object identified file with labels with supervised learned data.
            + labelled data with unidentified objects.

"""
from __future__ import print_function, division
import os
import time
import cv2
import tensorflow as tf
import keras
import keras_utils, keras_preprocessing, keras_applications
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Input, Dense, ZeroPadding2D, MaxPool2D, AveragePooling2D, Dropout, Flatten, merge, Reshape, \
    Activation
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

# sci libs
import numpy as np
import scipy
import scipy.misc

# Utility libraries
from tqdm import tqdm
import matplotlib.pyplot as plt

print("Begin processing!.................")
# One final thing , class names -> known objects on Mars. Total = 25
class_names = {'apxs', 'apxs cal target', 'chemcam cal target',
               'chemin inlet open', 'drill', 'drill holes', 'drt front', 'drt side', 'ground',
               'horizon', 'inlet', 'mahli', 'mahli cal target',
               'mastcam', 'mastcam cal target', 'observation tray',
               'portion box', 'portion tube', 'portion tube opening', 'rems uv sensor', 'rover rear deck',
               'scoop', 'sun', 'turret', 'wheel'
               }

# Process my ish!
IMAGES = np.array([])
instances = []
LABELS = []


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

        # print("to know that I am here. with count = ", count)
        lbl0 = "".join(img[1:])  # now extract the labels => should they be numeric?
        # print("extracted labels: content of lbl0: ", lbl0 + "   count:  ", count)

        # if line is empty
        # end of file is reached
        if not line:
            break
        else:
            # Extract image
            img1 = "./datasets/" + img0  # add the full path of image, missing from file,
            # print("full image_path: ", img1)
            # decided to go with, cv2.imread(), which saves as numpy matrix like we like.
            img2 = cv2.imread(img1)
            img2 = img2 / 255

            img3 = image.load_img(img1, target_size=(224, 224))
            img3 = image.img_to_array(img3)
            # img3 = np.expand_dims(img3, axis=0)
            img3 = preprocess_input(img3)

            # list of images.
            instances.append(img3)
            train_x = np.array(instances)
            train_x = preprocess_input(train_x)

            # Now process the labels.
            lbl0 = int(lbl0)  # now lbl0 has integer form of the label.
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

# Part-2 - Data processing done.
# Modelling begins.
# -----------------------------------------------------------------------
batch_size = 16
image_size = 224
num_class = 25

# loading the pre-trained model
# vgg = VGG16(include_top=False, input_shape=(image_size, image_size, 3), pooling='avg', weights='imagenet')
# vgg.summary()
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
print(base_model.summary())
base_model.trainable = False
print(base_model.summary())

# Creating the model template
model = keras.models.Sequential()
# Adding the vgg convolutional base model
model.add(base_model)

# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
# Summary of current model
print(model.summary())

model.compile(loss='categorical_crossentropy',
          optimizer=SGD(lr=1e-3),
          metrics=['accuracy'])


# # Start the training process
# model.fit(train_data, train_labels, validation_split=0.33, batch_size=batch_size, epochs=50, verbose=2)
# converting target variable to array
# train_y = np.asarray(train_labels)

# creating training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)


preds = model.predict(train_x)
# features = myModel.predict(train_x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])

# fitting the model
history = model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_valid, Y_valid))
len(model.trainable_variables)


initial_epochs = 10
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)



preds = model.predict(train_x)


# decode the results into a list of tuples (class, description, probability)
 print('Predicted:', decode_predictions(preds, top=3)[0])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# Print at the end.
print("End of Processing! ................")
