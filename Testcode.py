# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:11:48 2018

@author: guita
"""



# =============================================================================
# Import necessary library files
# =============================================================================

import cv2
import numpy as np
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.models import Model
import os, os.path
import pandas as pd
import sys
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.preprocessing import image

# =============================================================================
# Avoid Truncated Error
# =============================================================================

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



# =============================================================================
# Set Directories
# =============================================================================
model_weight_path='./weights/fulltrainweightsfile3.h5'
train_dataset_dir = "/media/rose/Windows/Final-Models"
test_dataset_dir = "/media/rose/Windows/Validation"


# =============================================================================
# Set variables
# =============================================================================

Trainable_layers_Number = 14
TrainBatchSize = 10
TestBatchSize = 10
sgd = SGD(lr=1e-2, decay=1e-6,momentum=0.9, nesterov=True)
adam = Adam(lr=0.001)


# =============================================================================
# Import Resnet Model and create newModel and loadweights
# =============================================================================
from resnet50 import ResNet50, preprocess_input
model = ResNet50()
# model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
desiredOutputs = model.get_layer('flatten').output
# desiredOutputs = Dropout(0.2)(desiredOutputs)
# desiredOutputs = Dense(1000)(desiredOutputs)
# desiredOutputs = Activation('relu')(desiredOutputs)
desiredOutputs = Dense(10)(desiredOutputs)
desiredOutputs = Activation('softmax')(desiredOutputs)
partialModel = Model(model.inputs,desiredOutputs)

partialModel.summary()

partialModel.load_weights(model_weight_path)

# =============================================================================
# Configure layers as trainable
# =============================================================================

# for layer in partialModel.layers[:-Trainable_layers_Number]:
#     layer.trainable = False
#
# for layer in partialModel.layers[-Trainable_layers_Number:]:
#     layer.trainable = True

for layer in partialModel.layers:
    layer.trainable = False
    print(layer.name,layer.trainable)

# =============================================================================
# Compile the Model
# =============================================================================
partialModel.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])




def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
print(get_model_memory_usage(TrainBatchSize,partialModel))

# =============================================================================
# List of Images directory and the list of label of the Images
# =============================================================================
# =============================================================================
# Create List for the label of the phonemodel
# =============================================================================

image_path_list = []
image_label_list = []
valid_image_extensions = [".jpg", ".jpeg"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

#create a list all files in directory and
#append files with a vaild extention to image_path_list

for phonemodel in os.listdir(os.path.join(test_dataset_dir)):
    for file in os.listdir(os.path.join(test_dataset_dir, phonemodel)):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue

#        image_path_list.append(os.path.join(test_dataset_dir,phonemodel, file))
        image_label_list.append(phonemodel)
        
        
predict_dir = '/media/rose/Windows/Final-Models/iPhone 4'	
for file in os.listdir(predict_dir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue

        image_path_list.append(os.path.join(predict_dir, file))
        

print(len(image_label_list))

# =============================================================================
#
# =============================================================================
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

encoder = LabelEncoder()
encoder.fit(image_label_list)

Labeldataset = encoder.transform(image_label_list)
Labeldataset = np_utils.to_categorical(Labeldataset)
# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================



# =============================================================================
# for image_path,image_label in zip(image_path_list, Labeldataset):
#     print(image_label)
#     im = image.load_img(image_path, target_size = (224,224))
#     img = image.img_to_array(im)
#     img=preprocess_input(img)
#     im_data = np.expand_dims(img, axis=0)
#     
#     print(partialModel.predict(im_data))
#     plt.imshow(im)
#     plt.show()
#     
# 
# =============================================================================
count = 0
total_count = 0
for image_path in image_path_list:
    total_count += 1
    im = image.load_img(image_path, target_size = (224,224))
    img = image.img_to_array(im)
    img=preprocess_input(img)
    im_data = np.expand_dims(img, axis=0)
    
    prediction = partialModel.predict(im_data)

    
    class_label = encoder.inverse_transform(prediction.argmax())
    if prediction.argmax() != 5:
        count += 1
        print(encoder.classes_)
        print(prediction)
        plt.imshow(im)
        plt.show()
    
    print(class_label, total_count, count)
 