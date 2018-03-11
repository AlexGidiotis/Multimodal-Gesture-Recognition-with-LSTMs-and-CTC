import random
import time
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

	def __init__(self,
		minibatch_size,
		numfeats_skeletal,
		numfeats_speech,
		maxlen,
		nb_classes,
		dataset,
		val_split=0.2,
		absolute_max_sequence_len=35):

		# Currently is only 2 files per batch.
		self.minibatch_size = minibatch_size
		# Maximum length of data sequence.
		self.maxlen = maxlen
		# 39 mel frequency feats.
		self.numfeats_speech = numfeats_speech
		# 22 skeletal feats.
		self.numfeats_skeletal = numfeats_skeletal
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

		self.dataset = dataset

		if self.dataset == 'train':
			self.in_audio_dir = '../data/train_audio'
			self.in_file_skeletal = '../data/Training_set_skeletal.csv'
		elif self.dataset == 'val':
			self.in_audio_dir = '../data/val_audio'
			self.in_file_skeletal = '../data/Validation_set_skeletal.csv'

		self.load_dataset()


	# Loads and preprocesses the dataset and splits it into training and validation set. 
	# This runs at initiallization.
	def load_dataset(self):
		print 'Loading data...'

		if self.dataset == 'train':
			train_lab_file = '../data/training_oov.csv'
		elif self.dataset == 'val':
			train_lab_file = '../data/validation.csv'

		
		labs = pd.read_csv(train_lab_file)
		self.labs = labs


		self.df_s = pd.read_csv(self.in_file_skeletal)
		self.df_s = self.normalize_data()


		file_list = os.listdir(self.in_audio_dir)
		file_list = sorted([int(re.findall('audio_(\d+).csv',file_name)[0]) for file_name in file_list])

		if self.dataset == 'train':
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
		else:
			self.val_list = file_list
			self.val_size = len(self.val_list)


	def get_size(self,train):
		"""
		"""

		if train:
			return self.train_size
		else:
			return self.val_size


	def get_file_list(self,train):
		if train:
			return self.train_list
		else:
			return self.val_list


	# Normalize the data to have zero mean and unity variance.
	# Called at initiallization.
	def normalize_data(self):

		data = self.df_s[['lh_v','rh_v','le_v','re_v','lh_dist_rp',
		'rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d',
		'lh_shc_d','rh_shc_d','le_shc_d','re_shc_d','lh_hip_ang',
		'rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang']].as_matrix().astype(float)
		norm_data = preprocessing.scale(data)
		norm_df = pd.DataFrame(norm_data,
			columns=['lh_v','rh_v','le_v','re_v','lh_dist_rp',
			'rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d',
			'lh_shc_d','rh_shc_d','le_shc_d','re_shc_d','lh_hip_ang',
			'rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang',
			'rh_el_ang'])
		norm_df['file_number'] = self.df_s['file_number']

		return norm_df


	# each time a batch (list of file ids) is requested from train/val/test
	# Basic function that returns a batch of training or validation data.
	# Returns a dictionary with the required fields for CTC training and a dummy output variable.
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
		X_data_a = np.ones([size, self.maxlen, self.numfeats_speech])
		labels_a = np.ones([size, self.absolute_max_sequence_len])
		input_length_a = np.zeros([size, 1])
		label_length_a = np.zeros([size, 1])
		X_data_s = np.ones([size, self.maxlen, self.numfeats_skeletal])
		labels_s = np.ones([size, self.absolute_max_sequence_len])
		input_length_s = np.zeros([size, 1])
		label_length_s = np.zeros([size, 1])


		# Read batch.
		for i in range(len(batch)):
			file = batch[i]
			audio_file_name = 'audio_' + str(file) + '.csv'
			audio_file_path = os.path.join(self.in_audio_dir,audio_file_name)

			vf_a = pd.read_csv(audio_file_path).drop(['file_number'],axis=1)
			if set(['39', '40']).issubset(vf_a.columns):
				vf_a = vf_a.drop(['39','40'],axis=1)

			# Downsample by 5 the audio.
			vf_a = vf_a.iloc[::5, :].reset_index(drop=True)

			vf_s = self.df_s[self.df_s['file_number'] == file]

			# Audio
			gest_seq_a = vf_a.as_matrix().astype(float)
			gest_seq_a = sequence.pad_sequences([gest_seq_a], 
				maxlen=self.maxlen, 
				padding='post',
				truncating='post',
				dtype='float32')
			# Skeletal
			gest_seq_s = vf_s[['lh_v','rh_v','le_v','re_v','lh_dist_rp',
			'rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d',
			'lh_shc_d','rh_shc_d','le_shc_d','re_shc_d','lh_hip_ang',
			'rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang']].as_matrix().astype(float)

			gest_seq_s = sequence.pad_sequences([gest_seq_s],
				maxlen=self.maxlen,
				padding='post',
				truncating='post',
				dtype='float32')

			
			lab_seq = self.labs[self.labs['Id'] == file]
			lab_seq = np.array([int(lab) for lab in lab_seq['Sequence'].values[0].split()]).astype('float32')


			# If a sequence is not found insert a blank example and pad.
			if lab_seq.shape[0] == 0:
				lab_seq = sequence.pad_sequences([self.blank_label],
					maxlen=(self.absolute_max_sequence_len),
					padding='post',
					value=-1)

				
				labels_s[i, :] = lab_seq
				labels_a[i, :] = lab_seq
				label_length_s[i] = 1
				label_length_a[i] = 1
			# Else use the save the returned variables.
			else:
				X_data_a[i, :, :] = gest_seq_a

				# Ignore empty examples
				try:
					X_data_s[i, :, :] = gest_seq_s
				except:
					print 'blank'

				label_length_a[i] = lab_seq.shape[0]
				label_length_s[i] = lab_seq.shape[0]
				lab_seq = sequence.pad_sequences([lab_seq],
					maxlen=(self.absolute_max_sequence_len),
					padding='post',
					value=-1)
				labels_a[i, :] = lab_seq
				labels_s[i, :] = lab_seq


			input_length_a[i] = (X_data_a[i].shape[0] - 2)
			input_length_s[i] = (X_data_s[i].shape[0] - 2)

		# Returned values: a dictionary with 4 values
		#	the_input_audio: audio data sequence
		#	the_input_skeletal: skeletal data sequence
		#	the labels: label sequence
		#	input_length: length of data sequence
		#	label_length: length of label sequence
		# an array of zeros
		#	outputs: dummy vector of zeros required for keras training
		inputs = {'the_input_audio': X_data_a,
				  'the_input_skeletal': X_data_s,
				  'the_labels': labels_a,
				  'input_length': input_length_a,
				  'label_length': label_length_a,
				  }
		outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

		return (inputs, outputs)


	# Get the next training batch and update index. 
	# Called by the generator during training.
	def next_train(self):
		while 1:
			ret = self.get_batch(train=True)


			self.train_index += self.minibatch_size
			if self.train_index >= self.train_size:
				self.train_index = 0
			yield ret


	# Get the next validation batch and update index. 
	# Called by the generator during validation.
	def next_val(self):
		while 1:
			ret = self.get_batch(train=False)


			self.val_index += self.minibatch_size
			if self.val_index >= self.val_size:
				self.val_index = 0
			yield ret


	# Save model and weights on epochs end.
	# Callback at the end of each epoch.
	def on_epoch_end(self, epoch, logs={}):
		self.train_index = 0
		self.val_index = 0

		random.shuffle(self.train_list)
		random.shuffle(self.val_list)

		
		model_json = self.model.to_json()
		with open("multimodal_ctc_blstm_model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights("multimodal_ctc_blstm_weights.h5")

		print "Saved model to disk"

