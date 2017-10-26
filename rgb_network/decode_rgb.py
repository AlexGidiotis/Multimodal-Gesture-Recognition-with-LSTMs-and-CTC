import os
import re
import random
import time
from operator import itemgetter
from itertools import groupby

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, add, Lambda, AlphaDropout
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Input
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.constraints import maxnorm
from keras.initializers import RandomUniform
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import model_from_json


#========================================================= Global variables =================================================
test_path = '/home/alex/Documents/Data/test_up_body'
validation_path = '/home/alex/Documents/Data/validation_up_body'

maxlen = 1900
img_dim = 48
nb_classes = 22
absolute_max_sequence_len = 28
stamp = 'cnn_lstm'
minibatch_size = 2
nb_epoch = 100


#=========================================================== Definitions ====================================================
# The data generator will yield batches of data to the training algorithm.
class DataGenerator(callbacks.Callback):

	def __init__(self,
		minibatch_size,
		img_dim,
		maxlen,
		nb_classes,
		data_path,
		absolute_max_sequence_len=28):

		# Currently is only 2 files per batch.
		self.minibatch_size = minibatch_size
		# Maximum length of data sequence.
		self.maxlen = maxlen
		# Dimensionality of the images.
		self.img_dim = img_dim
		# Max number of labels per label sequence.
		self.absolute_max_sequence_len = absolute_max_sequence_len
		# Actually 1-21 classes and 22 is the blank label and 0 is oov.
		self.nb_classes = nb_classes
		# INdexing variable
		self.index = 0
		# The path where the data files are saved.
		self.data_path = data_path
		# Blank model to use.
		self.blank_label = np.array([self.nb_classes - 1])

		self.load_dataset()


	# Reads the filelist, shuffles it and splits into training and validation set. Also loads the lab file.
	def load_dataset(self):
		self.file_list = sorted(os.listdir(self.data_path))
		self.val_size = len(self.file_list)


		#Make sure that train and validation lists have an even length to avoid mini-batches of size 
		val_mod_by_batch_size = self.val_size % self.minibatch_size


		if val_mod_by_batch_size != 0:
			del self.file_list[-val_mod_by_batch_size:]
			self.val_size -= val_mod_by_batch_size

		return


	# Return file list.
	def get_file_list(self):
		return self.file_list


	# Return sizes.
	def get_size(self):
		return self.val_size


	# each time a batch (list of file ids) is requested from the train/val set
	def get_batch(self):
		# number of files in batch is 2 (Cannot support more than this with one GPU)
		#file_list = self.file_list
		#index = self.index

		# Get the batch.
		try:
			batch = self.file_list[self.index:(self.index + self.minibatch_size)]
		except:
			batch = self.file_list[self.index:]

		size = len(batch)

		# Initialize the variables to be returned.
		X_data = np.ones([size, self.maxlen, self.img_dim, self.img_dim, 1],
			dtype='float32')
		labels = np.ones([size, self.absolute_max_sequence_len])
		input_length = np.zeros([size, 1])
		label_length = np.zeros([size, 1])

		# Read batch.
		for i in range(len(batch)):
			file = batch[i]
			file_path = os.path.join(self.data_path,file)
			# This part of the file name is the number.
			file_num = int(file[6:11])
			# Load data file.
			gest_seq = np.load(file_path).astype(float)
			# Pad data sequence to max length.
			gest_seq = sequence.pad_sequences([gest_seq],
				maxlen=self.maxlen,
				padding='post',
				truncating='post',
				dtype='float32')


	
			X_data[i, :, :] = gest_seq
			lab_seq = np.array([1])
			label_length[i] = lab_seq.shape[0]
			labels[i, :] = lab_seq
			input_length[i] = (X_data[i].shape[0] - 2)
		
		# Normalize data to have unit variance.
		X_data /= 255.

		# Returned values: a dictionary with 4 values
		#	the_input: data sequence
		#	the labels: label sequence
		#	input_length: length of data sequence
		#	label_length: length of label sequence

		#	outputs: dummy vector of zeros required for keras training
		inputs = {'the_input': X_data,
				  'the_labels': labels,
				  'input_length': input_length,
				  'label_length': label_length,
				  }

		outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

		return (inputs, outputs)


	# Get the next training batch and update index. Called by the generator.
	def next_batch(self):
		while 1:
			ret = self.get_batch()
			self.index += self.minibatch_size
			if self.index >= self.val_size:
				self.index = 0
			yield ret


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]

	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss


# Loads a previously saved model to resume training.
# Returns: the compiled model.
def load_model():
	json_file = open(stamp + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(stamp + '.h5')

	print("Loaded model from disk")

	adam = Adam(lr=0.00008,
		clipvalue=0.5)

	y_pred = model.get_layer('softmax').output

	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
		optimizer=adam)

	pred_model = Model(inputs=model.input,
					outputs=model.get_layer('softmax').output)

	return pred_model


# Gets a batch of predictions and decodes it into predicted sequence.
# THe decoding here is best path with threshold.
def decode_batch(pred_out,f_list):
	# Map gesture codes to classes.
	map_gest = {0:"oov", 1:"VA", 2:"VQ", 3:"PF", 4:"FU", 5:"CP", 6:"CV", 7:"DC", 8:"SP", 9:"CN", 10:"FN", 11:"OK", 12:"CF", 13:"BS", 14:"PR", 15:"NU", 
				16:"FM", 17:"TT",  18:"BN",  19:"MC", 20:"ST", 21:"sil"}

	# These files are problematic during decoding.
	ignore_list = [228,298,299,300,303,304,334,343,373,375]

	# Write the output to .mlf
	of = open('rgb_ctc_recout.mlf', 'w')
	of.write('#!MLF!#\n')

	out = pred_out
	ret = []
	for j in range(out.shape[0]):
		out_prob = list(np.max(out[j, 2:],1))
		out_best = list(np.argmax(out[j, 2:],1))
		# Filter the probabilities to get the most confident predictions.
		'''
		for p,s in zip(out_prob,out_best):
			if p < 0.88:
				out_prob.remove(p)
				out_best.remove(s)
		'''
		out_best = [k for k, g in groupby(out_best)]

		outstr = [map_gest[i] for i in out_best]
		ret.append(outstr)

		f_num = int(f_list[j][6:11])

		if f_num in ignore_list:
			continue

		fileNum = str(format(f_num, '05'))
		fileName = 'Sample' + fileNum
		of.write('"*/%s.rec"\n' %fileName)
		for cl in outstr:
			of.write('%s\n' %cl)
		of.write('.\n') 

	of.close()

	return ret


#========================================================== Main function ===================================================
# Choose between validation and test mode. No difference between 
# validation and test data, just different paths.
#mode = raw_input('Choose test or validation: ')
mode = 'validation'

print mode

if mode == 'test':
	data_path = test_path
elif mode == 'validation':
	data_path = validation_path

data_gen = DataGenerator(minibatch_size=minibatch_size,
	img_dim=img_dim,
	maxlen=maxlen,
	nb_classes=nb_classes,
	data_path=data_path)


model = load_model()
model.summary()

print 'Making predictions...'
predictions = model.predict_generator(generator=data_gen.next_batch(),
	steps=data_gen.get_size()/minibatch_size,
	verbose=1)

print predictions.shape
f_list = data_gen.get_file_list()

print 'Decoding...'
decoded_res = decode_batch(predictions, f_list)