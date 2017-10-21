# Trains a Bidirectional RNN with LSTM to recognise continuous gesture sequences from audio input.
# THis is a word level recognition network.
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
		absolute_max_sequence_len=125):

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
		# Actually 1-42 classes and 43 is the blank label and 0 is oov.
		self.nb_classes = nb_classes
		# Blank model to use.
		self.blank_label = np.array([self.nb_classes - 1])

		self.load_dataset()

	# Loads and preprocesses the dataset and splits it into training and validation set.
	# The loaded data should be in a csv file with 41 columns. Columns 0-38 are the MFCC features. 
	# Column 39 is the audio file number and column 40 is the label column.
	def load_dataset(self):
		# THe audio data path.
		in_file = '/home/alex/Documents/Python/multimodal_gesture_recognition/speech_blstm/Training_set_audio_labeled.csv'

		self.df = pd.read_csv(in_file,
			header=None)

		# Zero mean and unity variance normalization.
		self.df = self.normalize_data()

		# Create and shuffle file list.
		# Column 39 is the file number.
		file_list = self.df[39].unique().tolist()

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
		# Column 39 has the filename and column 40 the labels.
		data = self.df.drop([39,40], axis=1).as_matrix().astype(float)

		norm_data = preprocessing.scale(data)

		norm_df = pd.DataFrame(norm_data)

		norm_df[39] = self.df[39]
		norm_df[40] = self.df[40]

		return norm_df

	# This method converts label sequences to word level label sequences.
	# Input: lab_seq: nd array of class label sequence.
	# Returns: lab_seq: nd array of word level label sequence.
	def sent_2_words(self,lab_seq):

		'''
		This dicts will not be used
		
		class_dict = {0:"oov", 1:"VA", 2:"VQ", 3:"PF", 4:"FU", 5:"CP", 6:"CV", 7:"DC", 8:"SP", 9:"CN", 10:"FN", 11:"OK", 12:"CF", 13:"BS", 14:"PR",
					 15:"NU", 16:"FM", 17:"TT",  18:"BN",  19:"MC", 20:"ST", 21:"sil"}

		word_dict = {0:"oov", 1:"vattene", 2:"vieni", 3:"qui", 4:"perfetto", 5:"e'", 6:"un", 7:"furbo", 8:"che", 9:"due", 10:"palle",
					11:"vuoi", 12:"vanno", 13:"d'accordo", 14:"sei", 15:"pazzo", 16:"cos'hai", 17:"combinato", 18:"non", 19:"me", 20:"ne", 21:"frega",
					22:"niente", 23:"ok", 24:"cosa", 25:"ti", 26:"farei", 27:"basta", 28:"le", 29:"prendere", 30:"ce", 31:"n'e", 32:"piu",
					33:"ho", 34:"fame", 35:"tanto", 36:"tempo", 37:"fa", 38:"buonissimo", 39:"si", 40:"sono", 41:"messi", 42:"stufo" , 43:"sil"}
		'''

		# Here we map classes to keyword sequences.
		class_2_words = {0:[0.0], 1:[1.0], 2:[2.0,3.0], 3:[4.0], 4:[5.0,6.0,7.0], 5:[8.0,9.0,10.0], 6:[8.0,11.0], 7:[12.0,13.0], 8:[14.0,15.0],
						9:[16.0,17.0], 10:[18.0,19.0,20.0,21.0,22.0], 11:[23.0], 12:[24.0,25.0,26.0], 13:[27.0], 14:[28.0,11.0,29.0], 
						15:[18.0,30.0,31.0,32.0], 16:[33.0,34.0], 17:[35.0,36.0,37.0], 18:[38.0], 19:[39.0,40.0,41.0,13.0], 20:[40.0,42.0], 21:[41.0]}

		# Go through the sequence of classes and convert it to sequence of keywords.
		# Empty list to insert the new sequence.
		new_seq = []
		for lab in lab_seq:
			# Append the list of keywords for each class.
			new_seq = new_seq + class_2_words[lab]

		# Back to np array.
		lab_seq = np.asarray(new_seq)

		return lab_seq


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
			vf = self.df[self.df[39] == file]
			# Downsample by 5 the audio.
			vf = vf.iloc[::5, :].reset_index(drop=True)
			# SElect and pad data sequence to max length.
			gest_seq = vf.drop([39,40],axis=1).as_matrix().astype(float)
			gest_seq = sequence.pad_sequences([gest_seq],
				maxlen=self.maxlen,
				padding='post',
				truncating='post',
				dtype='float32')
			
			# Create the label vector. Ignores the blanks.
			lab_seq = vf[vf[40] != 0][40].unique().astype('float32')
			index = np.argwhere(lab_seq==0)
			lab_seq = np.delete(lab_seq, index)
			lab_seq = self.sent_2_words(lab_seq)

			# Insert oovs between gesture labels. (did not improve things)
			# lab_seq = np.insert(lab_seq, slice(1, None), 0)

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
		model_json = self.model.to_json()
		with open("sp_ctc_lstm_model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights("sp_ctc_lstm_weights.h5")

		print "Saved model to disk"

#============================================================== CTC LOSS ===================================================================
# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]

	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss

# ====================================================== MAIN ==========================================================================================
minibatch_size = 2
val_split = 0.25
maxlen = 1900
nb_classes = 44
nb_epoch = 150
numfeats = 39

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
lstm_1 = Bidirectional(LSTM(500,
	name='blstm_1',
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

#===================================================== LABELS ==========================================================================================
labels = Input(name='the_labels',
	shape=[data_gen.absolute_max_sequence_len],
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

model = Model(input=[input_data, labels, input_length, label_length],
	output=[loss_out])

# Optimizer.
# Clipping the gradients to have smaller values makes the training smoother.
adam = Adam(lr=0.0001,
	clipvalue=0.5)

# Resume training.
if load_previous == 'yes':
	json_file = open('sp_ctc_lstm_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("sp_ctc_lstm_weights_best.h5")

	adam = Adam(lr=0.0001,
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
filepath="sp_ctc_lstm_weights_best.h5"
checkpoint = ModelCheckpoint(filepath,
	monitor='val_loss',
	verbose=1,
	save_best_only=True,
	save_weights_only=True,
	mode='auto')

# ====================================================== TRAIN =========================================================================================
print 'Start training.'
start_time = time.time()

model.fit_generator(generator=data_gen.next_train(),
	steps_per_epoch=(data_gen.get_size(train=True)/minibatch_size),
	epochs=nb_epoch,
	validation_data=data_gen.next_val(),
	validation_steps=(data_gen.get_size(train=False)/minibatch_size),
	callbacks=[earlystopping, checkpoint, data_gen])

#================================================================================================================================

end_time = time.time()
print "--- Training time: %s seconds ---" % (end_time - start_time)
