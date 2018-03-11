# Trains a Bidirectional RNN with LSTM to recognise continuous gesture sequences from audio input.
import time
import random
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

#====================================================== DATA GENERATOR =================================================================================
# Data generator that will provide training and testing with data. Works with mini batches of audio feat files.
# The data generator is called using the next_train() and next_val() methods.

# Class constructor to initialize the datagenerator object.
class DataGenerator(keras.callbacks.Callback):

	def __init__(self,
		minibatch_size,
		numfeats,
		maxlen,
		val_split,
		nb_classes,
		absolute_max_sequence_len=28):

		# Currently is only 2 files per batch.
		self.minibatch_size = minibatch_size
		# Maximum length of data sequence.
		self.maxlen = maxlen
		# 39 mel frequency feats.
		self.numfeats = numfeats
		# Size of the validation set.
		self.val_split = val_split
		# Max number of labels per label sequence.
		self.absolute_max_sequence_len = absolute_max_sequence_len
		# INdexing variables
		self.train_index = 0
		self.val_index = 0
		# Actually 1-20 classes and 21 is the blank label.
		self.nb_classes = nb_classes
		# Blank model to use.
		self.blank_label = np.array([self.nb_classes - 1])

		self.load_dataset()

	# Loads and preprocesses the dataset and splits it into training and validation set.
	# THe input data should be in csv file with the required 24 columns. We are using 22 extracted skeletal features.
	def load_dataset(self):
		# The input file path.
		in_file = 'Training_set_skeletal.csv'
		train_lab_file = '../training.csv'

		labs = pd.read_csv(train_lab_file)
		self.labs = labs
		self.df = pd.read_csv(in_file)
		self.df = self.df[['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp',
			'lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d',
			'le_shc_d','re_shc_d','lh_hip_ang','rh_hip_ang','lh_shc_ang',
			'rh_shc_ang','lh_el_ang','rh_el_ang', 'file_number']]

		# Zero mean and unity variance normalization.
		self.df = self.normalize_data()

		# Create and shuffle file list.
		file_list = self.df['file_number'].unique().tolist()

		random.seed(10)
		random.shuffle(file_list)

		# SPlit to training and validation set.
		split_point = int(len(file_list) * (1 - self.val_split))
		self.train_list, self.val_list = file_list[:split_point], file_list[split_point:]
		self.train_size = len(self.train_list)
		self.val_size = len(self.val_list)

#=====================================================================================================================================================
	#Make sure that train and validation lists have an even length to avoid mini-batches of size 1
		train_mod_by_batch_size = self.train_size % self.minibatch_size

		if train_mod_by_batch_size != 0:
			del self.train_list[-train_mod_by_batch_size:]
			self.train_size -= train_mod_by_batch_size

		val_mod_by_batch_size = self.val_size % self.minibatch_size

		if val_mod_by_batch_size != 0:
			del self.val_list[-val_mod_by_batch_size:]
			self.val_size -= val_mod_by_batch_size
#=====================================================================================================================================================
	# Return sizes.
	def get_size(self,train):
		if train:
			return self.train_size
		else:
			return self.val_size

	# Normalize the data to have zero mean and unity variance.
	def normalize_data(self):

		data = self.df[['lh_v','rh_v','le_v','re_v','lh_dist_rp',
		'rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d',
		'lh_shc_d','rh_shc_d','le_shc_d','re_shc_d','lh_hip_ang',
		'rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang',
		'rh_el_ang']].as_matrix().astype(float)

		norm_data = preprocessing.scale(data)

		norm_df = pd.DataFrame(norm_data,
			columns=['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp',
			'lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d',
			'le_shc_d','re_shc_d','lh_hip_ang','rh_hip_ang','lh_shc_ang',
			'rh_shc_ang','lh_el_ang','rh_el_ang'])

		norm_df['file_number'] = self.df['file_number']

		return norm_df

	# each time a batch (list of file ids) is requested from train/val/test
	def get_batch(self, train):
		# number of files in batch is 2
		
		# Select train or validation mode.
		if train:
			file_list = self.train_list
			index = self.train_index
		else:
			file_list = self.val_list
			index = self.val_index

		# Get the batch.
		try:
			batch = file_list[index:(index + self.minibatch_size)]
		except:
			batch = file_list[index:]

		size = len(batch)

		# INitialize the variables to be returned.
		X_data = np.ones([size, self.maxlen, self.numfeats])
		labels = np.ones([size, self.absolute_max_sequence_len])
		input_length = np.zeros([size, 1])
		label_length = np.zeros([size, 1])

		# Read batch.
		for i in range(len(batch)):
			file = batch[i]
			vf = self.df[self.df['file_number'] == file]
			
			
			# SElect and pad data sequence to max length.
			gest_seq = vf[['lh_v','rh_v','le_v','re_v','lh_dist_rp',
			'rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d',
			'lh_shc_d','rh_shc_d','le_shc_d','re_shc_d','lh_hip_ang',
			'rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang']].as_matrix().astype(float)
			gest_seq = sequence.pad_sequences([gest_seq],
				maxlen=self.maxlen,
				padding='post',
				truncating='post',
				dtype='float32')
			
			# Create the label vector.
			lab_seq = self.labs[self.labs['Id'] == file]
			lab_seq = lab_seq['Sequence'].values

			# If a sequence is not found insert a blank example and pad.
			if lab_seq.shape[0] == 0:
				lab_seq = sequence.pad_sequences([self.blank_label],
					maxlen=(self.absolute_max_sequence_len),
					padding='post',
					value=-1)
				labels[i, :] = lab_seq
				label_length[i] = 1
			# Else use the save the returned variables.
			else:
				X_data[i, :, :] = gest_seq
				lab_seq = lab_seq[0].split()
				lab_seq = np.array([int(lab) for lab in lab_seq]).astype('float32')
				label_length[i] = lab_seq.shape[0]
				lab_seq = sequence.pad_sequences([lab_seq],
					maxlen=(self.absolute_max_sequence_len),
					padding='post',
					value=-1)
				labels[i, :] = lab_seq

			input_length[i] = (X_data[i].shape[0] - 2)

		# Returned values: a dictionary with 4 values
		#	the_input: data sequence
		#	the labels: label sequence
		#	input_length: length of data sequence
		#	label_length: length of label sequence
		# an array of zeros
		#	outputs: dummy vector of zeros required for keras training
		inputs = {'the_input': X_data,
				  'the_labels': labels,
				  'input_length': input_length,
				  'label_length': label_length,
				  }

		outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
		return (inputs, outputs)

	# Get the next training batch and update index. Called by the generator.
	def next_train(self):
		while 1:
			ret = self.get_batch(train=True)
			self.train_index += self.minibatch_size
			if self.train_index >= self.train_size:
				self.train_index = 0
			yield ret

	# Get the next validation batch and update index. Called by the generator.
	def next_val(self):
		while 1:
			ret = self.get_batch(train=False)
			self.val_index += self.minibatch_size
			if self.val_index >= self.val_size:
				self.val_index = 0
			yield ret

	# Save model and weights on epochs end.
	def on_epoch_end(self, epoch, logs={}):
		self.train_index = 0
		self.val_index = 0

		random.shuffle(self.train_list)
		random.shuffle(self.val_list)


		model_json = self.model.to_json()
		with open("sk_ctc_lstm_model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights("sk_ctc_lstm_weights.h5")

		print "Saved model to disk"


def ctc_lambda_func(args):
	"""
	"""

	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]

	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss


if __name__ == '__main__':

	minibatch_size = 2
	val_split = 0.2
	maxlen = 1900
	nb_classes = 22
	nb_epoch = 500
	numfeats = 20

	K.set_learning_phase(1)  # all new operations will be in train mode from now on

	uni_initializer = RandomUniform(minval=-0.05,
		maxval=0.05,
		seed=47)

	# Can load a previous model to resume training. 'yes'.
	load_previous = raw_input("Load previous model? ")
	# Create data generator object.
	data_gen = DataGenerator(minibatch_size=minibatch_size,
		numfeats=numfeats,
		maxlen=maxlen,
		val_split=val_split,
		nb_classes=nb_classes)

	# Shape of the network input.
	input_shape = (maxlen, numfeats)

	input_data = Input(name='the_input',
		shape=input_shape,
		dtype='float32')

	# Add noise as a regularizer. Increasing the standard deviation of the noise makes the inputs noisier.
	input_noise = GaussianNoise(stddev=0.5)(input_data)

	# BLSTM layers
	# Block 1
	# We add dropout to the inputs but not to the recurrent connections.
	# We use kernel constraints to make the weights smaller.
	lstm_1 = Bidirectional(LSTM(300, name='blstm_1',
		activation='tanh',
		recurrent_activation='hard_sigmoid',
		recurrent_dropout=0.0,
		dropout=0.6, 
		kernel_constraint=maxnorm(3),
		kernel_initializer=uni_initializer,
		return_sequences=True),
		merge_mode='concat')(input_noise)

	# Block 2
	# We add dropout to the inputs but not to the recurrent connections.
	# We use kernel constraints to make the weights smaller.
	lstm_2 = Bidirectional(LSTM(300,
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
	res_block = layers.add([lstm_1, lstm_2])

	# More dropout will help regularization.
	dropout_1 = Dropout(0.6,
		name='dropout_layer_1')(res_block)

	# Softmax output layers
	# Block 3
	# Predicts a class probability distribution at every time step.
	inner = Dense(nb_classes,
		name='dense_1',
		kernel_initializer=uni_initializer)(dropout_1) 
	y_pred = Activation('softmax',
		name='softmax')(inner)

	Model(input=[input_data],
		output=y_pred).summary()


	labels = Input(name='the_labels',
		shape=[data_gen.absolute_max_sequence_len],
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

	# Optimizer.
	# Clipping the gradients to have smaller values makes the training smoother.
	adam = Adam(lr=0.0001,
		decay=1e-5,
		clipvalue=0.5)

	# Resume training.
	if load_previous == 'yes':
		json_file = open('sk_ctc_lstm_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights("sk_ctc_lstm_weights_best.h5")

		adam = Adam(lr=0.0001,
			decay=1e-5,
			clipvalue=0.5)
		print("Loaded model from disk")

	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
		optimizer=adam)

	# Early stopping to avoid overfitting.
	earlystopping = EarlyStopping(monitor='val_loss',
		patience=20,
		verbose=1)
	# Checkpoint to save the weights with the best validation accuracy.
	filepath="sk_ctc_lstm_weights_best.h5"
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

