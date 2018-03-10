
import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import groupby
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import RMSprop
import keras.callbacks
from keras.layers import Input, Lambda
from keras.models import Model
import itertools
from sklearn import preprocessing
#====================================================== DATA GENERATOR =================================================================================
# Uses generator functions to supply train/test with
# data.
# The data generator is called using the next_batch()method.

# Class constructor to initialize the datagenerator object.
class DataGenerator(keras.callbacks.Callback):

	def __init__(self, minibatch_size, numfeats_speech, numfeats_skeletal, maxlen,nb_classes, test_set,
				 absolute_max_sequence_len=28):
		# Currently is only 2 files per batch.
		self.minibatch_size = minibatch_size
		# Maximum length of data sequence.
		self.maxlen = maxlen
		# 39 mel frequency feats.
		self.numfeats_speech = numfeats_speech
		# 22 skeletal feats.
		self.numfeats_skeletal = numfeats_skeletal
		# Max number of labels per label sequence.
		self.absolute_max_sequence_len = absolute_max_sequence_len
		# INdexing variable
		self.index = 0
		# Actually 1-20 classes and 21 is the blank label.
		self.nb_classes = nb_classes
		self.test = test_set
		self.load_dataset()

	# Loads and preprocesses the dataset and splits it into training and validation set. 
	# This runs at initiallization.
	def load_dataset(self):
		print 'Loading data...'
		# The input files.
		if self.test == 'test':
			in_file_audio = '/home/alex/Documents/Python/multimodal_gesture_recognition/speech_blstm/Testing_set_audio.csv'
			in_file_skeletal = '../skeletal_network/Validation_set_skeletal.csv'
		elif self.test == 'train':
			in_file_audio = '/home/alex/Documents/Python/multimodal_gesture_recognition/speech_blstm/Training_set_audio_labeled.csv'
			in_file_skeletal = '../skeletal_network/Training_set_skeletal.csv'
		else:
			in_file_audio = '../audio_network/Final_set_audio.csv'
			in_file_skeletal = '../skeletal_network/final_set_skeletal.csv'
		# Read the inputs.
		self.df_a = pd.read_csv(in_file_audio, header=None)
		self.df_s = pd.read_csv(in_file_skeletal)

		# Zero mean and unity variance normalization.
		self.df_a = self.normalize_data('audio')
		self.df_s = self.normalize_data('skeletal')

		file_list = self.df_a[39].unique().tolist()

		self.test_list = file_list
		self.test_size = len(self.test_list)

#=====================================================================================================================================================
#Make sure that train and validation lists have an even length to avoid mini-batches of size 1
		test_mod_by_batch_size = self.test_size % self.minibatch_size

		if test_mod_by_batch_size != 0:
			del self.test_list[-test_mod_by_batch_size:]
			self.test_size -= test_mod_by_batch_size

#=====================================================================================================================================================
	# Return size.
	# Called in order to get the test set size.
	def get_size(self):
		return self.test_size
	# Return file list.
	def get_file_list(self):
		return self.test_list

	# Normalize the data to have zero mean and unity variance.
	# Called at initiallization.
	def normalize_data(self,stream):
		# Normalize audio.
		if stream == 'audio':
			
			if self.test == 'train':
				data = self.df_a.drop([39,40], axis=1).as_matrix().astype(float)
			else:
				data = self.df_a.drop([39], axis=1).as_matrix().astype(float)

			norm_data = preprocessing.scale(data)

			norm_df = pd.DataFrame(norm_data)

			norm_df[39] = self.df_a[39]
			if self.test == 'train':
				norm_df[40] = self.df_a[40]

		# Normalize skeletal.
		elif stream == 'skeletal':
			data = self.df_s[['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d','le_shc_d','re_shc_d',
							'lh_hip_ang','rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang']].as_matrix().astype(float)

			norm_data = preprocessing.scale(data)

			norm_df = pd.DataFrame(norm_data, columns=['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d','le_shc_d','re_shc_d',
					'lh_hip_ang','rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang'])

			norm_df['file_number'] = self.df_s['file_number']

		return norm_df

	# each time a batch (list of file ids) is requested from test
	# Basic function that returns a batch of testing data.
	# Returns a dictionary with the required fields for the network.
	def get_batch(self):
		# number of files in batch 
		file_list = self.test_list
		index = self.index

		try:
			batch = file_list[index:(index + minibatch_size)]
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

		# REad the mini batch.
		for i in range(len(batch)):
			file = batch[i]
			print file
			vf_a = self.df_a[self.df_a[39] == file]
			# Downsample by 5 the audio.
			vf_a = vf_a.iloc[::5, :].reset_index(drop=True)
			vf_s = self.df_s[self.df_s['file_number'] == file]

			# SElect and pad data sequence to max length.
			# Audio
			
			if self.test == 'train':
				gest_seq_a = vf_a.drop([39,40], axis=1).as_matrix().astype(float)
			else:
				gest_seq_a = vf_a.drop([39], axis=1).as_matrix().astype(float)

			gest_seq_a = sequence.pad_sequences([gest_seq_a], maxlen=self.maxlen, padding='post', truncating='post', dtype='float32')

			# Skeletal
			gest_seq_s = vf_s[['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d','le_shc_d','re_shc_d',
							'lh_hip_ang','rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang']].as_matrix().astype(float)
			gest_seq_s = sequence.pad_sequences([gest_seq_s], maxlen=self.maxlen, padding='post', truncating='post', dtype='float32')
			
			# Save the returned variables.
			X_data_a[i, :, :] = gest_seq_a
			# The skeletal stream is sometimes inconsistent.
			try:
				X_data_s[i, :, :] = gest_seq_s
			except:
				pass
			# Label sequence length and label sequence are not used here so we just set to zero.
			lab_seq = np.array([1])
			label_length_a[i] = lab_seq.shape[0]
			labels_a[i, :] = lab_seq
			# The length of the data sequence. (same for both streams)
			input_length_a[i] = (X_data_a[i].shape[0] - 2)

		# Returned values: a dictionary with 4 values
		#	the_input_audio: audio data sequence
		#	the_input_skeletal: skeletal data sequence
		#	the labels: label sequence
		#	input_length: length of data sequence
		#	label_length: length of label sequence
		inputs = {'the_input_audio': X_data_a,
				  'the_input_skeletal': X_data_s,
				  'the_labels': labels_a,
				  'input_length': input_length_a,
				  'label_length': label_length_a,
				  }

		return (inputs)

	# Get the next batch and update index. 
	# Called by the generator.
	def next_batch(self):
		while 1:
			ret = self.get_batch()
			self.index += self.minibatch_size
			if self.index >= self.test_size:
				self.index = 0
			yield ret

#================================================================= CTC LOSS ========================================================================
# The CTC loss is only used in training. We just need it to compile the model here.
# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]

	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss

#================================================================== DECODE ==========================================================================
# Gets a batch of predictions and decodes it into predicted sequence.
# THe decoding here is best path with threshold.
def decode_batch(pred_out,f_list):
	# Map gesture codes to classes.
	map_gest = {0:"oov", 1:"VA", 2:"VQ", 3:"PF", 4:"FU", 5:"CP", 6:"CV", 7:"DC", 8:"SP", 9:"CN", 10:"FN", 11:"OK", 12:"CF", 13:"BS", 14:"PR", 15:"NU", 
				16:"FM", 17:"TT",  18:"BN",  19:"MC", 20:"ST", 21:"sil"}

	# These files are problematic during decoding.
	ignore_list = [228,298,299,300,303,304,334,343,373,375]

	# Write the output to .mlf
	of = open('final_ctc_recout.mlf', 'w')
	of.write('#!MLF!#\n')

	out = pred_out
	ret = []
	for j in range(out.shape[0]):
		out_prob = list(np.max(out[j, 2:],1))
		out_best = list(np.argmax(out[j, 2:],1))
		# Filter the probabilities to get the most confident predictions.
		
		for p,s in zip(out_prob,out_best):
			if p < 0.97:
				out_prob.remove(p)
				out_best.remove(s)

		out_best = [k for k, g in itertools.groupby(out_best)]

		outstr = [map_gest[i] for i in out_best]
		ret.append(outstr)

		f_num = f_list[j]

		if int(f_num) in ignore_list:
			continue

		fileNum = str(format(f_num, '05'))
		fileName = 'Sample'+fileNum
		of.write('"*/%s.rec"\n' %fileName)
		for cl in outstr:
			of.write('%s\n' %cl)
		of.write('.\n') 

	of.close()

	return ret

# ============================================================== MAIN ========================================================================

map_gest = {0:"oov", 1:"VA", 2:"VQ", 3:"PF", 4:"FU", 5:"CP", 6:"CV", 7:"DC", 8:"SP", 9:"CN", 10:"FN", 11:"OK", 12:"CF", 13:"BS", 14:"PR", 15:"NU", 
				16:"FM", 17:"TT",  18:"BN",  19:"MC", 20:"ST", 21:"sil"}
# ====================================================== LOAD THE SAVED NETWORK ==============================================================
minibatch_size = 2
maxlen = 1900
nb_classes = 22
nb_epoch = 100
numfeats_speech = 39
numfeats_skeletal = 20

K.set_learning_phase(0)  # all new operations will be in test mode from now on

test_set = raw_input('select train or test: ')
data_gen = DataGenerator(minibatch_size=minibatch_size, numfeats_speech=numfeats_speech, numfeats_skeletal=numfeats_skeletal, maxlen=maxlen, nb_classes=nb_classes, test_set=test_set)

# Shape of the network inputs.
input_shape_a = (maxlen, numfeats_speech)
input_shape_s = (maxlen, numfeats_skeletal)

# Input layers for the audio and skeletal.
input_data_a = Input(name='the_input_audio', shape=input_shape_a, dtype='float32')
input_data_s = Input(name='the_input_skeletal', shape=input_shape_s, dtype='float32')

# load json and create model
json_file = open('multimodal_ctc_blstm_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("multimodal_ctc_lstm_weights_best.h5")
print("Loaded model from disk")

y_pred = loaded_model.get_layer('softmax').output
labels = Input(name='the_labels', shape=[data_gen.absolute_max_sequence_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

rmsprop = RMSprop(lr=0.001, clipnorm=5)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=rmsprop)

pred_model = Model(inputs=loaded_model.input,
					outputs=loaded_model.get_layer('softmax').output)

predictions = pred_model.predict_generator(generator=data_gen.next_batch(), steps=data_gen.get_size()/minibatch_size)
f_list = data_gen.get_file_list()

decoded_res = decode_batch(predictions, f_list)
