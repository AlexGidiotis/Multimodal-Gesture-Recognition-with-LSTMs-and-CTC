import os
import re
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, add, Lambda
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.constraints import maxnorm
from keras.initializers import RandomUniform
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam

#========================================================= Global variables =================================================
train_path = '/home/alex/Documents/Data/training_up_body_images'
validation_path = '/home/alex/Documents/Data/validation_up_body_images'

seq_len,img_x,img_y = 1200,64,64
nb_classes = 22
input_shape = (seq_len,img_x,img_y,1)
absolute_max_sequence_len = 28
stamp = 'cnn_lstm'


#=========================================================== Definitions ====================================================
# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]

	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss


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

	Model(inputs=[input_layer], outputs=y_pred).summary()

	labels = Input(name='the_labels',
		shape=[absolute_max_sequence_len],
		dtype='float32')
	input_length = Input(name='input_length',
		shape=[1],
		dtype='int64')
	label_length = Input(name='label_length',
		shape=[1],
		dtype='int64')

	# ================================================= COMPILE THE MODEL ==================================================================================
	# Keras doesn't currently support loss funcs with extra parameters
	# so CTC loss is implemented in a lambda layer
	# The extra parameters required for ctc is a label sequence (list), input sequence length, and label sequence length.
	loss_out = Lambda(ctc_lambda_func,
		output_shape=(1,),
		name="ctc")([y_pred, labels, input_length, label_length])

	model = Model(inputs=[input_layer, labels, input_length, label_length],
		outputs=[loss_out])

	# Optimizer.
	# Clipping the gradients to have smaller values makes the training smoother.
	adam = Adam(lr=0.0001,
		clipvalue=0.5)

	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
		optimizer=adam)

	# Save the model.
	model_json = model.to_json()
	with open(stamp + ".json", "w") as json_file:
	    json_file.write(model_json)

	# Early stopping to avoid overfitting.
	earlystopping = EarlyStopping(monitor='val_loss',
		patience=20,
		verbose=1)
	# Checkpoint to save the weights with the best validation accuracy.
	best_model_path = stamp + ".h5"
	checkpoint = ModelCheckpoint(best_model_path,
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=True,
		mode='auto')

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