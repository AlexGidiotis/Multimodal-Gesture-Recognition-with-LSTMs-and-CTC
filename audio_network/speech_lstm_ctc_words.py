import time
import itertools
import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Input, Lambda, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from keras.optimizers import RMSprop, Adam
import keras.callbacks
from keras.models import load_model
from keras.models import model_from_json
from keras.constraints import maxnorm
from keras.layers.noise import GaussianNoise
from keras import layers
from keras.initializers import RandomUniform

from data_generator import DataGenerator
from losses import ctc_lambda_func


def build_model(maxlen,
	numfeats,
	nb_classes,
	lab_seq_len,
	resume_training):
	"""
	"""

	K.set_learning_phase(1)

	uni_initializer = RandomUniform(minval=-0.05,
		maxval=0.05,
		seed=47)

	
	input_shape = (maxlen, numfeats)
	input_data = Input(name='the_input',
		shape=input_shape,
		dtype='float32')


	input_noise = GaussianNoise(stddev=0.5)(input_data)

	
	lstm_1 = Bidirectional(LSTM(500,
		name='blstm_1',
		activation='tanh',
		recurrent_activation='hard_sigmoid',
		recurrent_dropout=0.0,
		dropout=0.4, 
		kernel_constraint=maxnorm(3),
		kernel_initializer=uni_initializer,
		return_sequences=True),
		merge_mode='concat')(input_noise)


	lstm_2 = Bidirectional(LSTM(500,
		name='blstm_2',
		activation='tanh',
		recurrent_activation='hard_sigmoid',
		recurrent_dropout=0.0,
		dropout=0.5,
		kernel_constraint=maxnorm(3),
		kernel_initializer=uni_initializer,
		return_sequences=True),
		merge_mode='concat')(lstm_1)

	res_block = layers.add([lstm_1, lstm_2])


	dropout_1 = Dropout(0.5,
		name='dropout_layer_1')(res_block)


	inner = Dense(nb_classes,
		name='dense_1',
		kernel_initializer=uni_initializer)(dropout_1) 
	y_pred = Activation('softmax',
		name='softmax')(inner)

	Model(input=[input_data],
		output=y_pred).summary()


	labels = Input(name='the_labels',
		shape=[lab_seq_len],
		dtype='float32')
	input_length = Input(name='input_length',
		shape=[1],
		dtype='int64')
	label_length = Input(name='label_length',
		shape=[1],
		dtype='int64')


	loss_out = Lambda(ctc_lambda_func,
		output_shape=(1,),
		name="ctc")([y_pred, labels, input_length, label_length])

	model = Model(input=[input_data, labels, input_length, label_length],
		output=[loss_out])


	adam = Adam(lr=0.0001,
		clipvalue=0.5)

	if resume_training == 'yes':
		json_file = open('sp_ctc_lstm_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights("sp_ctc_lstm_weights_best.h5")

		adam = Adam(lr=0.0001,
			clipvalue=0.5)
		print("Loaded model from disk")


	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
		optimizer=adam)

	return model


if __name__ == '__main__':

	minibatch_size = 2
	val_split = 0.2
	maxlen = 1900
	nb_classes = 44
	nb_epoch = 500
	numfeats = 39

	dataset='train'


	load_previous = raw_input("Load previous model? ")

	data_gen = DataGenerator(minibatch_size=minibatch_size,
		numfeats=numfeats,
		maxlen=maxlen,
		dataset=dataset,
		val_split=val_split,
		nb_classes=nb_classes)


	lab_seq_len = data_gen.absolute_max_sequence_len
	model = build_model(maxlen=maxlen,
		numfeats=numfeats,
		nb_classes=nb_classes,
		lab_seq_len=lab_seq_len,
		resume_training=load_previous)


	earlystopping = EarlyStopping(monitor='val_loss',
		patience=20,
		verbose=1)


	filepath="sp_ctc_lstm_weights_best.h5"
	checkpoint = ModelCheckpoint(filepath,
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=True,
		mode='auto')


	print 'Start training.'
	start_time = time.time()

	model.fit_generator(generator=data_gen.next_train(),
		steps_per_epoch=(data_gen.get_size(train=True)/minibatch_size),
		epochs=nb_epoch,
		validation_data=data_gen.next_val(),
		validation_steps=(data_gen.get_size(train=False)/minibatch_size),
		callbacks=[checkpoint, data_gen])


	end_time = time.time()
	print "--- Training time: %s seconds ---" % (end_time - start_time)
