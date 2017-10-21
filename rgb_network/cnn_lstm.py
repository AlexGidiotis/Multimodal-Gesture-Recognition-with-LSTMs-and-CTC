import os
import re
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, add
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.constraints import maxnorm
from keras.initializers import RandomUniform
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

#========================================================= Global variables =================================================
train_path = '/home/alex/Documents/Data/training_up_body_images'
validation_path = '/home/alex/Documents/Data/validation_up_body_images'

seq_len,img_x,img_y = 1200,64,64
nb_classes = 22
input_shape = (seq_len,img_x,img_y,1)
#=========================================================== Definitions ====================================================
def build_net():

	uni_initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=47)

	input_layer = Input(name='the_input', 
		shape=input_shape)

	# CNN Block 1
	drop1 = TimeDistributed(Dropout(0.5
		),name='drop_1')(input_layer)
	conv1 = TimeDistributed(Convolution2D(32, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3)),
		name='conv_1')(drop1)
	conv2 = TimeDistributed(Convolution2D(32, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3)),
		name='conv_2')(conv1)
	pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)),
		name='max_pool_1')(conv2)

	# CNN Block 2
	drop2 = TimeDistributed(Dropout(0.5),
		name='drop_2')(pool1)
	conv3 = TimeDistributed(Convolution2D(64, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3)),
		name='conv_3')(drop2)
	conv4 = TimeDistributed(Convolution2D(64, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform',
		kernel_constraint=maxnorm(3)),
		name='conv_4')(conv3)
	pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)),
		name='max_pool_2')(conv4)

	flat = TimeDistributed(Flatten(),name='flatten')(pool2)
	
	# LSTM Block 1
	# We add dropout to the inputs but not to the recurrent connections.
	# We use kernel constraints to make the weights smaller.
	lstm_1 = Bidirectional(LSTM(500, 
		name='blstm_1', 
		activation='tanh', 
		recurrent_activation='hard_sigmoid', 
		recurrent_dropout=0.0, 
		dropout=0.6, 
		kernel_constraint=maxnorm(3), 
		kernel_initializer=uni_initializer, 
		return_sequences=True), 
		merge_mode='concat')(flat)

	# LSTM Block 2
	# We add dropout to the inputs but not to the recurrent connections.
	# We use kernel constraints to make the weights smaller.
	lstm_2 = Bidirectional(LSTM(500,
		name='blstm_2', 
		activation='tanh', 
		recurrent_activation='hard_sigmoid', 
		recurrent_dropout=0.0, 
		dropout=0.6,
		kernel_constraint=maxnorm(3), 
		kernel_initializer=uni_initializer, 
		return_sequences=True), 
		merge_mode='concat')(lstm_1)
	# The block can be residual. Makes the training a little bit easier.
	res_block_1 = add([lstm_1, lstm_2],
		name='residual_1')

	
	# Softmax output layers
	# Dense Block
	# More dropout will help regularization.
	drop3 = Dropout(0.6,
		name='drop_3')(res_block_1)
	# Predicts a class probability distribution at every time step.
	inner = Dense(nb_classes,
		name='dense_1',
		kernel_initializer=uni_initializer)(drop3) 
	y_pred = Activation('softmax',
		name='softmax')(inner)

	Model(input=[input_layer], output=y_pred).summary()

	return model

#========================================================== Main function ===================================================
# Choose between train and test mode. No difference between train and test data, just different paths.
#mode = raw_input('Choose train or validation: ')
mode = 'train'
print mode

if mode == 'train':
	data_path = train_path
elif mode == 'validation':
	data_path = validation_path

model = build_net()