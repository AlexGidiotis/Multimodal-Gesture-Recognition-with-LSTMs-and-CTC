import random
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Input, Lambda, TimeDistributed, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import time
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from keras.optimizers import RMSprop, Adam, Nadam
import keras.callbacks
from sklearn import preprocessing
import itertools
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1,l2
from keras.constraints import maxnorm
from keras import layers

#====================================================== DATA GENERATOR =================================================================================
# Data generator that will provide training and testing with data. Works with mini batches of audio feat files.
# The data generator is called using the next_train() and next_val() methods.


# Class constructor to initialize the datagenerator object.
class DataGenerator(keras.callbacks.Callback):

	def __init__(self, minibatch_size, numfeats_skeletal, numfeats_speech,maxlen,val_split,nb_classes,
				 absolute_max_sequence_len=28):

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
		self.load_dataset()


	# Loads and preprocesses the dataset and splits it into training and validation set. 
	# This runs at initiallization.
	def load_dataset(self):
		print 'Loading data...'
		# The input files.
		in_file_audio = '/home/alex/Documents/Python/multimodal_gesture_recognition/speech_blstm/Training_set_audio_labeled.csv'
		in_file_skeletal = '/home/alex/Documents/Python/multimodal_gesture_recognition/Training_set_skeletal.csv'
		# Read the inputs.
		self.df_a = pd.read_csv(in_file_audio, header=None)
		self.df_s = pd.read_csv(in_file_skeletal)
		# Zero mean and unity variance normalization.
		self.df_a = self.normalize_data('audio')
		self.df_s = self.normalize_data('skeletal')


		# Create and shuffle file list. (The same for both skeletal and audio)
		file_list = self.df_a[39].unique().tolist()
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
	# Called in order to get the training and validation set sizes.
	def get_size(self,train):
		if train:
			return self.train_size
		else:
			return self.val_size


	# Normalize the data to have zero mean and unity variance.
	# Called at initiallization.
	def normalize_data(self,stream):
		# Normalize audio.
		if stream == 'audio':
			data = self.df_a.drop([39,40], axis=1).as_matrix().astype(float)
			norm_data = preprocessing.scale(data)
			norm_df = pd.DataFrame(norm_data)
			norm_df[39] = self.df_a[39]
			norm_df[40] = self.df_a[40]


		# Normalize skeletal.
		elif stream == 'skeletal':
			data = self.df_s[['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d','le_shc_d','re_shc_d',
							'lh_hip_ang','rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang','lh_dir','rh_dir']].as_matrix().astype(float)
			norm_data = preprocessing.scale(data)
			norm_df = pd.DataFrame(norm_data, columns=['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d','le_shc_d','re_shc_d',
					'lh_hip_ang','rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang','lh_dir','rh_dir'])
			norm_df['file_number'] = self.df_s['file_number']
			norm_df['labels'] = self.df_s['labels']

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
			vf_a = self.df_a[self.df_a[39] == file]
			# Downsample by 5 the audio.
			vf_a = vf_a.iloc[::5, :].reset_index(drop=True)
			vf_s = self.df_s[self.df_s['file_number'] == file]


			# SElect and pad data sequence to max length.
			# Audio
			gest_seq_a = vf_a.drop([39,40], axis=1).as_matrix().astype(float)
			gest_seq_a = sequence.pad_sequences([gest_seq_a], maxlen=self.maxlen, padding='post',truncating='post', dtype='float32')
			# Skeletal
			gest_seq_s = vf_s[['lh_v','rh_v','le_v','re_v','lh_dist_rp','rh_dist_rp','lh_hip_d','rh_hip_d','le_hip_d','re_hip_d','lh_shc_d','rh_shc_d','le_shc_d','re_shc_d',
							'lh_hip_ang','rh_hip_ang','lh_shc_ang','rh_shc_ang','lh_el_ang','rh_el_ang','lh_dir','rh_dir']].as_matrix().astype(float)
			gest_seq_s = sequence.pad_sequences([gest_seq_s], maxlen=self.maxlen, padding='post',truncating='post', dtype='float32')
			

			# Create the label vector. Ignores the blanks.(Same for skeletal and audio)
			lab_seq = vf_a[vf_a[40] != 0][40].unique().astype('float32')
			index = np.argwhere(lab_seq==0)
			lab_seq = np.delete(lab_seq, index)


			# If a sequence is not found insert a blank example and pad.
			if lab_seq.shape[0] == 0:
				lab_seq = sequence.pad_sequences([self.blank_label], maxlen=(self.absolute_max_sequence_len), padding='post', value=-1)
				
				labels_s[i, :] = lab_seq
				labels_a[i, :] = lab_seq
				label_length_s[i] = 1
				label_length_a[i] = 1
			# Else use the save the returned variables.
			else:
				X_data_a[i, :, :] = gest_seq_a
				X_data_s[i, :, :] = gest_seq_s
				label_length_a[i] = lab_seq.shape[0]
				label_length_s[i] = lab_seq.shape[0]
				lab_seq = sequence.pad_sequences([lab_seq], maxlen=(self.absolute_max_sequence_len), padding='post', value=-1)
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
		model_json = self.model.to_json()
		with open("multimodal_ctc_blstm_model.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights("multimodal_ctc_blstm_weights.h5")

		print "Saved model to disk"


#============================================================== CTC LOSS ==============================================================================
# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]


	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss


#=========================================================== FREEZES BIDIRECTIONAL LAYERS =============================================================
# Freeze the Bidirectional Layers
# The Bidirectional wrapper is buggy and does not support freezing. This is a workaround that freezes each bidirectional layer.
def layer_trainable(l, freeze, verbose=False, bidir_fix=True):
	l.trainable = freeze

	if bidir_fix:
		if type(l) == Bidirectional:
			l.backward_layer.trainable = not freeze
			l.forward_layer.trainable = not freeze

	if verbose:
		if freeze:
			action='Froze' 
		else :
			action='Unfroze'
		print("{} {}".format(action, l.name))


# ====================================================== MAIN ==========================================================================================
minibatch_size = 2
val_split = 0.25
maxlen = 1900
nb_classes = 22
nb_epoch = 150
numfeats_speech = 39
numfeats_skeletal = 22

# Can load a previous model to resume training. 'yes'
load_previous = raw_input("Load previous model? ")
# Create data generator object.
data_gen = DataGenerator(minibatch_size=minibatch_size, numfeats_skeletal=numfeats_skeletal,numfeats_speech=numfeats_speech, 
						maxlen=maxlen, val_split=val_split, nb_classes=nb_classes)

# Load pre-trained uni-modal networks.
skeletal_model_file = '/home/alex/Documents/Python/multimodal_gesture_recognition/skeletal_pretrain/sk_ctc_lstm_model.json'
skeletal_weights = '/home/alex/Documents/Python/multimodal_gesture_recognition/skeletal_pretrain/sk_ctc_lstm_weights_best.h5'
speech_model_file = '/home/alex/Documents/Python/multimodal_gesture_recognition/speech_pretrain/sp_ctc_lstm_model.json'
speech_weights = '/home/alex/Documents/Python/multimodal_gesture_recognition/speech_pretrain/sp_ctc_lstm_weights_best.h5'


json_file = open(skeletal_model_file, 'r')
skeletal_model_json = json_file.read()
json_file.close()
skeletal_model = model_from_json(skeletal_model_json)
# load weights into new model
skeletal_model.load_weights(skeletal_weights)


json_file = open(speech_model_file, 'r')
speech_model_json = json_file.read()
json_file.close()
speech_model = model_from_json(speech_model_json)
# load weights into new model
speech_model.load_weights(speech_weights)

#===================================================== BUILD THE MODEL =================================================================================
# Shape of the network inputs.
input_shape_a = (maxlen, numfeats_speech)
input_shape_s = (maxlen, numfeats_skeletal)

# Input layers for the audio and skeletal.
input_data_a = Input(name='the_input_audio', shape=input_shape_a, dtype='float32')
input_data_s = Input(name='the_input_skeletal', shape=input_shape_s, dtype='float32')

# Add zero-centered gaussian noise to the input. (For better regularization)
input_noise_a = GaussianNoise(stddev=0.5, name='gaussian_noise_a')(input_data_a)
input_noise_s = GaussianNoise(stddev=0.5, name='gaussian_noise_s')(input_data_s)


# Blocks 1(audio) run in parallel with 2(skeletal).
# Audio BLSTM
# kernel_constraint=maxnorm(10),
# Block 1
# Load the pre-trained layers and add them to the model.
blstm_1_a = speech_model.layers[2](input_noise_a)
blstm_2_a = speech_model.layers[3](blstm_1_a)
# Build residual connection.
res_a_1 = layers.add([blstm_1_a, blstm_2_a], name='speech_residual')
#dropout_a = speech_model.layers[5](res_a_1)

# Skeletal BLSTM
# Block 2
# Load the pre-trained layers and add them to the model.
blstm_1_s = skeletal_model.layers[2](input_noise_s)
blstm_2_s = skeletal_model.layers[3](blstm_1_s)
# Build residual connection.
res_s_1 = layers.add([blstm_1_s, blstm_2_s], name='skeletal_residual')
#dropout_s = skeletal_model.layers[5](res_s_1)


# Build the audio model and rename layers.
model_a = Model(input=[input_data_a], output=res_a_1)
model_a.layers[2].name='speech_blstm_1'
model_a.layers[3].name='speech_blstm_2'
#model_a.layers[5].name='speech_dropout'

# Build the skeletal model and rename layers.
model_s = Model(input=[input_data_s], output=res_s_1)
model_s.layers[2].name='skeletal_blstm_1'
model_s.layers[3].name='skeletal_blstm_2'
#model_s.layers[5].name='skeletal_dropout'


# attempt to freeze all Bidirectional layers.
# Bidirectional wrapper layer is buggy so we need to freeze the weights this way.
frozen_types = [Bidirectional]
# Go through layers for both networks and freeze the weights of Bidirectional layers.
for l_a,l_s in zip(model_a.layers,model_s.layers):
	if len(l_a.trainable_weights):
		if type(l_a) in frozen_types:
			layer_trainable(l_a, freeze=True, verbose=True)

	if len(l_s.trainable_weights):
		if type(l_s) in frozen_types:
			layer_trainable(l_s, freeze=True, verbose=True)


model_a.summary()
model_s.summary()


# Merge the parallel single modality networks.
merged = Merge([model_a, model_s], mode='concat')([res_a_1,res_s_1])

# Block 3
lstm_3 = Bidirectional(LSTM(100, name='blstm_2', activation='tanh', recurrent_activation='hard_sigmoid', recurrent_dropout=0.0, dropout=0.5,
			kernel_constraint=maxnorm(3), kernel_initializer='glorot_uniform', return_sequences=True), merge_mode='concat')(merged)


dropout_3 = Dropout(0.6, name='dropout_layer_3')(lstm_3)

# Softmax output layer
# Block 4
inner = Dense(nb_classes, name='dense_1', kernel_initializer='glorot_uniform')(dropout_3)
y_pred = Activation('softmax', name='softmax')(inner)

Model(input=[input_data_a,input_data_s], output=y_pred).summary()


#===================================================== LABELS ==========================================================================================
# These are also inputes needed for the CTC loss
labels = Input(name='the_labels', shape=[data_gen.absolute_max_sequence_len], dtype='float32')

input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


# ================================================= COMPILE THE MODEL ==================================================================================
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

# The complete model with the CTC loss.
model = Model(input=[input_data_a,input_data_s, labels, input_length, label_length], output=[loss_out])

# Optimizer.
adam = Adam(lr=0.0001, clipvalue=0.5)


# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

# Early stopping to avoid overfitting
earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
# Checkpoint to save the weights with the best validation accuracy.
filepath="multimodal_ctc_lstm_weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')


# ====================================================== TRAIN =========================================================================================
print 'Start training.'
start_time = time.time()

model.fit_generator(generator=data_gen.next_train(), steps_per_epoch=(data_gen.get_size(train=True)/minibatch_size),
					epochs=nb_epoch, validation_data=data_gen.next_val(), validation_steps=(data_gen.get_size(train=False)/minibatch_size),
					callbacks=[earlystopping, checkpoint, data_gen])


#================================================================================================================================

end_time = time.time()
print "--- Training time: %s seconds ---" % (end_time - start_time)
