# from Conv_DCFD_tf2 import *

import tensorflow as tf
import os
import cv2
import json
import numpy as np
import math
from Conv_DCFD_tf import *

trainDir = "ACDA/input"
modelName = "people_reg.h5"

img_size = (256,256,3) # (160,160)

AUTOTUNE = tf.data.AUTOTUNE
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


def get_images(trainDir):
	with open("shanghaitech_people_count.json") as json_file:
		label_dict = json.load(json_file)
	# print(label_dict)
	images = []
	labels = []
	for img_name in label_dict:
		img = cv2.imread(trainDir + "/" + img_name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, img_size[0:2])
		img = img.astype(np.float32) / 255


		images.append(img)
		labels.append((np.log(label_dict[img_name])-3.25)/(7.8-3.25))
	return np.array(images), np.array(labels)

X_train, y_train = get_images(trainDir)

# X_valid, y_valid = X_train[0:20], y_train[0:20]
# X_train, y_train = X_train[21::], y_train[21::]


if True: # with CONV_DCFD layers
	model = tf.keras.Sequential()
	# Add a convolutional layer with 32 filters and a kernel size of (3, 3)
	model.add(Conv_DCFD_tf(in_channels=3, out_channels=16, kernel_size=3, inter_kernel_size=5, stride=1, num_bases=6))
	model.add(tf.keras.layers.Activation('relu'))

	model.add(tf.keras.layers.MaxPooling2D((2, 2)))

	model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
	# model.add(Conv_DCFD_tf(in_channels=32, out_channels=64, kernel_size=3, inter_kernel_size=5, stride=1, num_bases=6))
	# model.add(tf.keras.layers.Activation('relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(512, activation='relu'))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(1))

else: # without CONV_DCFD layers
	model = tf.keras.Sequential()
	# Add a convolutional layer with 32 filters and a kernel size of (3, 3)
	model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=img_size))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(512, activation='relu'))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(1))





model.compile(optimizer='adam', loss='MAE')

print(X_train.shape, y_train.shape)

model.fit(X_train, y_train, epochs=1, batch_size=32)

error_sum = 0
error_squared = 0
count = 0
for img, real in zip(X_train, y_train):
	# image_og = cv2.resize(img, img_size[0:2])
	
	image = np.expand_dims(image_og, axis=0)

	res = model.predict(image)
	print(res)
	pred = float(np.exp(res * (7.8-3.25) + 3.25))
	print("real", real)
	real = float(np.exp(real * (7.8-3.25) + 3.25))
	print("real2", real)

	error_sum += abs(pred-real)
	error_squared += (pred-real)**2
	count += 1
	image_og = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print("real", real, "pred", pred, "mse", error_squared/count, "mae", error_sum/count)
	cv2.imshow("iog", image_og)
	if cv2.waitKey(0) == 27:
		break

# b = model
# import random
# model.save(modelName)
# model = load_model(modelName', custom_objects={'circular_loss': circular_loss})


"""
DCFD mse 66314.26354680157 mae 161.83250778990907

"""