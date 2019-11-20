#!/Users/albertolanaro/venv3/bin/python3
import sys
import os
import numpy as np
import cv2
import glob
from random import shuffle
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
def load_imgs(path):

	usr = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
	DEFAULT_SIZE = (30,30)
	imgs = []
	for i in usr:
		imgs.append([cv2.resize(cv2.imread(j, cv2.IMREAD_GRAYSCALE), DEFAULT_SIZE, interpolation=cv2.INTER_CUBIC) for j in glob.glob('Faces/' + i + '/*.png')])

	print('Number of users:', len(imgs))
	for i,j in enumerate(imgs):
		print('\tSamples for user %d: %d' % (i+1,len(j)))

	return imgs, usr

def create_labels(imgs):
	y = []
	for i in range(len(imgs)):
		y.append(i * np.ones(len(imgs[i])))
	y = np.hstack(y)

	return y

def reshape_for_keras(data):
	temp = np.vstack(data)
	
	return temp.reshape(temp.shape + (1,))

def train_test_split(data, test_size):
	rnd_index = np.arange(data.shape[0])
	shuffle(rnd_index)
	last_test_index = round(test_size * len(rnd_index))
	test_index =  rnd_index[:last_test_index]
	train_index = rnd_index[last_test_index:]

	return train_index, test_index

def create_model(data, n_classes):
	input_data = Input(shape=data.shape[1:])
	X = Conv2D(32, (7, 7), strides = (1, 1),padding = 'Same',name = 'conv0')(input_data)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	X = Conv2D(64, (4, 4), strides = (1, 1),padding = 'Same', name = 'conv1')(X)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((2, 2), name='max_pool1')(X)
	X = Dropout(0.4)(X)


	# CONV -> CONV -> BN -> MAXPOOL -> DropOut
	X = Conv2D(128, (4, 4), strides = (1,1),padding = 'Same', name = 'conv2')(X) 
	X = BatchNormalization()(X) 
	X = Activation('relu')(X)
	X = Conv2D(256, (2, 2), strides = (2, 2),padding = 'Valid', name = 'conv3')(X)
	X = BatchNormalization()(X) 
	X = Activation('relu')(X)

	X = MaxPooling2D((2, 2), name='max_pool2')(X)
	X = Dropout(0.4)(X)

	# FLATTEN -> Dense -> BN -> DropOut
	X = Flatten()(X)
	X = Dense(257,  name='fc1')(X)
	X = Activation('relu')(X)
	X = BatchNormalization()(X)
	X = Dropout(0.4)(X)

	#Dense -> BN -> DropOut
	X = Dense(126, name='fc2')(X)
	X = Activation('relu')(X)
	X = BatchNormalization()(X)
	X = Dropout(0.6)(X)

	output_layer = Dense(n_classes, activation='softmax', name='fcf')(X)
	# Model for prediction
	cl = Model(input_data, output_layer)
	cl.summary()

	cl.compile(loss=keras.losses.categorical_crossentropy, 
			   optimizer='Adam', 
			   metrics=['accuracy'])
	# Model for feature extraction
	return cl

def train_and_evaluate(cl, x_train, y_train, x_test, y_test, n_classes):
	batch_size = 64
	epochs = 10

	y_trainCNN = keras.utils.to_categorical(y_train, n_classes)
	y_testCNN = keras.utils.to_categorical(y_test, n_classes)

	cl.fit(x_train, y_trainCNN, batch_size=batch_size, epochs=epochs, verbose=1)
	pred = cl.predict(x_test)
	y_pred = np.argmax(pred, axis=1)
	conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))
	accuracy = 1-np.sum(np.abs(y_pred - y_test)) / len(y_test)

	print('Confusion matrix:\n', conf_matrix)
	print('Test accuracy:', accuracy)

def save_model(model):
	model_json = model.to_json()
	with open("trained_models/trained_model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("trained_models/model_weights.h5")
	print("Saved model to disk")

