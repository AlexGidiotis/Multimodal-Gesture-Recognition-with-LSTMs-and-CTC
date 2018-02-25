
import time
import random
import os
import re

import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.preprocessing import sequence
import keras.callbacks

#====================================================== DATA GENERATOR =================================================================================
# Data generator that will provide training and testing with data. Works with mini batches of audio feat files.
# The data generator is called using the next_train() and next_val() methods.

# Class constructor to initialize the datagenerator object.
class DataGenerator(keras.callbacks.Callback):
	"""
	"""

	def __init__(self,
		minibatch_size,
		numfeats,
		maxlen,
		val_split,
		nb_classes,
		absolute_max_sequence_len=150):
		"""
		"""

		self.minibatch_size = minibatch_size
		self.maxlen = maxlen
		self.numfeats = numfeats
		self.val_split = val_split
		self.absolute_max_sequence_len = absolute_max_sequence_len
		self.train_index = 0
		self.val_index = 0
		self.nb_classes = nb_classes
		self.blank_label = np.array([self.nb_classes - 1])

		self.in_dir = '../data/train_audio'

		self.build_dataset()

	# Loads and preprocesses the dataset and splits it into training and validation set.
	# The loaded data should be in a csv file with 41 columns. Columns 0-38 are the MFCC features. 
	# Column 39 is the audio file number and column 40 is the label column.
	def build_dataset(self):
		"""
		"""

		train_lab_file = '../data/training_oov.csv'

		labs = pd.read_csv(train_lab_file)
		self.labs = labs


		file_list = os.listdir(self.in_dir)
		file_list = sorted([int(re.findall('audio_(\d+).csv',file_name)[0]) for file_name in file_list])
		random.seed(10)
		random.shuffle(file_list)
		

		split_point = int(len(file_list) * (1 - self.val_split))
		self.train_list, self.val_list = file_list[:split_point], file_list[split_point:]
		self.train_size = len(self.train_list)
		self.val_size = len(self.val_list)


		#Make sure that train and validation lists have an even length to avoid mini-batches of size 1
		train_mod_by_batch_size = self.train_size % self.minibatch_size

		if train_mod_by_batch_size != 0:
			del self.train_list[-train_mod_by_batch_size:]
			self.train_size -= train_mod_by_batch_size

		val_mod_by_batch_size = self.val_size % self.minibatch_size

		if val_mod_by_batch_size != 0:
			del self.val_list[-val_mod_by_batch_size:]
			self.val_size -= val_mod_by_batch_size

		return

	# Return sizes.
	def get_size(self,train):
		"""
		"""

		if train:
			return self.train_size
		else:
			return self.val_size


	# This method converts label sequences to word level label sequences.
	# Input: lab_seq: nd array of class label sequence.
	# Returns: lab_seq: nd array of word level label sequence.
	def sent_2_words(self,lab_seq):

		"""
		This dicts will not be used
		
		class_dict = {0:"oov", 1:"VA", 2:"VQ", 3:"PF", 4:"FU", 5:"CP", 6:"CV", 7:"DC", 8:"SP", 
					9:"CN", 10:"FN", 11:"OK", 12:"CF", 13:"BS", 14:"PR",
					15:"NU", 16:"FM", 17:"TT",  18:"BN",  19:"MC", 20:"ST", 21:"sil"}

		word_dict = {0:"oov", 1:"vattene", 2:"vieni", 3:"qui", 4:"perfetto", 5:"e'", 6:"un", 7:"furbo", 
				8:"che", 9:"due", 10:"palle", 11:"vuoi", 12:"vanno", 13:"d'accordo", 14:"sei", 15:"pazzo", 
				16:"cos'hai", 17:"combinato", 18:"non", 19:"me", 20:"ne", 21:"frega",
				22:"niente", 23:"ok", 24:"cosa", 25:"ti", 26:"farei", 27:"basta", 28:"le", 29:"prendere", 
				30:"ce", 31:"n'e", 32:"piu", 33:"ho", 34:"fame", 35:"tanto", 36:"tempo", 37:"fa", 
				38:"buonissimo", 39:"si", 40:"sono", 41:"messi", 42:"stufo" , 43:"sil"}
		"""

		class_2_words = {0:[0.0], 1:[1.0], 2:[2.0,3.0], 3:[4.0], 4:[5.0,6.0,7.0], 5:[8.0,9.0,10.0], 6:[8.0,11.0], 7:[12.0,13.0], 8:[14.0,15.0],
						9:[16.0,17.0], 10:[18.0,19.0,20.0,21.0,22.0], 11:[23.0], 12:[24.0,25.0,26.0], 13:[27.0], 14:[28.0,11.0,29.0], 
						15:[18.0,30.0,31.0,32.0], 16:[33.0,34.0], 17:[35.0,36.0,37.0], 18:[38.0], 19:[39.0,40.0,41.0,13.0], 20:[40.0,42.0], 21:[43.0]}


		new_seq = []
		for lab in lab_seq:
			new_seq = new_seq + class_2_words[lab]

		lab_seq = np.asarray(new_seq)

		return lab_seq


	# each time a batch (list of file ids) is requested from train/val/test
	def get_batch(self, train):
		"""
		"""

		if train:
			file_list = self.train_list
			index = self.train_index
		else:
			file_list = self.val_list
			index = self.val_index


		try:
			batch = file_list[index:(index + self.minibatch_size)]
		except:
			batch = file_list[index:]

		size = len(batch)


		X_data = np.ones([size, self.maxlen, self.numfeats])
		labels = np.ones([size, self.absolute_max_sequence_len])
		input_length = np.zeros([size, 1])
		label_length = np.zeros([size, 1])


		for i in range(len(batch)):
			file = batch[i]
			file_name = 'audio_' + str(file) + '.csv'
			file_path = os.path.join(self.in_dir,file_name)
			vf = pd.read_csv(file_path).drop(['file_number'],axis=1)

			if set(['39', '40']).issubset(vf.columns):
				vf = vf.drop(['39','40'],axis=1)

			# Downsample by 5 the audio.
			vf = vf.iloc[::5, :].reset_index(drop=True)


			gest_seq = vf.as_matrix().astype(float)

			gest_seq = sequence.pad_sequences([gest_seq],
				maxlen=self.maxlen,
				padding='post',
				truncating='post',
				dtype='float32')
			

			lab_seq = self.labs[self.labs['Id'] == file]
			lab_seq = np.array([int(lab) for lab in lab_seq['Sequence'].values[0].split()]).astype('float32')

			lab_seq = self.sent_2_words(lab_seq)

			# If a sequence is not found insert a blank example and pad.
			if lab_seq.shape[0] == 0:
				lab_seq = sequence.pad_sequences([self.blank_label],
					maxlen=(self.absolute_max_sequence_len),
					padding='post',
					value=-1)
				labels[i, :] = lab_seq
				label_length[i] = 1
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
		"""
		"""

		while 1:
			ret = self.get_batch(train=True)
			self.train_index += self.minibatch_size
			if self.train_index >= self.train_size:
				self.train_index = 0
			yield ret

	# Get the next validation batch and update index. Called by the generator.
	def next_val(self):
		"""
		"""

		while 1:
			ret = self.get_batch(train=False)
			self.val_index += self.minibatch_size
			if self.val_index >= self.val_size:
				self.val_index = 0
			yield ret

	# Save model and weights on epochs end.
	def on_epoch_end(self, epoch, logs={}):
		"""
		"""

		model_json = self.model.to_json()
		with open("sp_ctc_lstm_model.json", "w") as json_file:
			json_file.write(model_json)

		self.model.save_weights("sp_ctc_lstm_weights.h5")

		print "Saved model to disk"
