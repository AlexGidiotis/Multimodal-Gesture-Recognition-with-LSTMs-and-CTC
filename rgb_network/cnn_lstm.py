import os
import re
import random
import time
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
train_path = '/home/alex/Documents/Data/training_up_body'
train_lab_file = '../training.csv'

maxlen = 1900
img_dim = 48
nb_classes = 22
absolute_max_sequence_len = 28
stamp = 'cnn_lstm'
minibatch_size = 2
val_split = 0.2
nb_epoch = 100


#=========================================================== Definitions ====================================================
# The data generator will yield batches of data to the training algorithm.
class DataGenerator(callbacks.Callback):

	def __init__(self,
		minibatch_size,
		img_dim,
		maxlen,
		val_split,
		nb_classes,
		data_path,
		lab_file,
		absolute_max_sequence_len=28):

		# Currently is only 2 files per batch.
		self.minibatch_size = minibatch_size
		# Maximum length of data sequence.
		self.maxlen = maxlen
		# Dimensionality of the images.
		self.img_dim = img_dim
		# Size of the validation set.
		self.val_split = val_split
		# Max number of labels per label sequence.
		self.absolute_max_sequence_len = absolute_max_sequence_len
		# INdexing variables
		self.train_index = 0
		self.val_index = 0
		# Actually 1-21 classes and 22 is the blank label and 0 is oov.
		self.nb_classes = nb_classes
		# The path where the data files are saved.
		self.data_path = data_path
		# The .csv files with the labels.
		self.lab_file = lab_file
		# Blank model to use.
		self.blank_label = np.array([self.nb_classes - 1])

		self.load_dataset()


	# Reads the filelist, shuffles it and splits into training and validation set. Also loads the lab file.
	def load_dataset(self):
		labs = pd.read_csv(self.lab_file)
		self.labs = labs
		file_list = sorted(os.listdir(self.data_path))

		random.seed(10)
		random.shuffle(file_list)

		# Split to training and validation set.
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
		if train:
			return self.train_size
		else:
			return self.val_size


	# each time a batch (list of file ids) is requested from the train/val set
	def get_batch(self, train):
		# number of files in batch is 2 (Cannot support more than this with one GPU)
		
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


			# Create the label vector.
			lab_seq = self.labs[self.labs['Id'] == file_num]
			lab_seq = lab_seq['Sequence'].values
			# If a sequence is not found insert a blank example and pad.
			if lab_seq.shape[0] == 0:
				lab_seq = sequence.pad_sequences([self.blank_label],
					maxlen=(self.absolute_max_sequence_len),
					padding='post',
					value=-1)
				labels[i, :] = lab_seq
				label_length[i] = 1
			# Else save the returned variables.
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


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]

	ctc_batch_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return ctc_batch_loss


# Builds the network.
# Returns: the compiled model.
def build_net():
	uni_initializer = RandomUniform(minval=-0.05,
		maxval=0.05,
		seed=47)

	input_shape = (maxlen,img_dim,img_dim,1)

	input_layer = Input(name='the_input', 
		shape=input_shape)

	# CNN Block 1
	drop1 = TimeDistributed(Dropout(0.2),
		name='drop_1')(input_layer)
	conv1 = TimeDistributed(Convolution2D(8, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform'),
		name='conv_1')(drop1)
	#conv1 = TimeDistributed(BatchNormalization(),
	#	name='bn_1')(conv1)
	conv2 = TimeDistributed(Convolution2D(8, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform'),
		name='conv_2')(conv1)
	#conv2 = TimeDistributed(BatchNormalization(),
	#	name='bn_2')(conv2)
	pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)),
		name='max_pool_1')(conv2)

	# CNN Block 2
	drop2 = TimeDistributed(Dropout(0.2),
		name='drop_2')(pool1)
	conv3 = TimeDistributed(Convolution2D(16, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform'),
		name='conv_3')(drop2)
	#conv3 = TimeDistributed(BatchNormalization(),
	#	name='bn_3')(conv3)
	conv4 = TimeDistributed(Convolution2D(16, (3,3),
		activation='relu', 
		padding='valid',
		kernel_initializer='lecun_uniform'),
		name='conv_4')(conv3)
	#conv4 = TimeDistributed(BatchNormalization(),
	#	name='bn_4')(conv4)
	pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)),
		name='max_pool_2')(conv4)

	flat = TimeDistributed(Flatten(),name='flatten')(pool2)
	
	# LSTM Block 1
	lstm_1 = Bidirectional(LSTM(300, 
		name='blstm_1', 
		activation='tanh', 
		recurrent_activation='hard_sigmoid', 
		recurrent_dropout=0.0, 
		dropout=0.2, 
		kernel_initializer=uni_initializer, 
		return_sequences=True), 
		merge_mode='concat')(flat)
	lstm_1 = BatchNormalization(name='bn_5')(lstm_1)

	# LSTM Block 2
	lstm_2 = Bidirectional(LSTM(300,
		name='blstm_2', 
		activation='tanh', 
		recurrent_activation='hard_sigmoid', 
		recurrent_dropout=0.0, 
		dropout=0.2, 
		kernel_initializer=uni_initializer, 
		return_sequences=True), 
		merge_mode='concat')(lstm_1)
	lstm_2 = BatchNormalization(name='bn_6')(lstm_2)
	res_block_1 = add([lstm_1, lstm_2],
		name='residual_1')

	# Dense Block
	drop3 = Dropout(0.2,
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


	# Keras doesn't currently support loss funcs with extra parameters
	# so CTC loss is implemented in a lambda layer
	# The extra parameters required for ctc is a label sequence (list), 
	# input sequence length, and label sequence length.
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

	return model


# Loads a previously saved model to resume training.
# Returns: the compiled model.
def load_model():
	json_file = open(stamp + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(stamp + '.h5')

	adam = Adam(lr=0.0001,
		clipvalue=0.5)
	print("Loaded model from disk")

	y_pred = model.get_layer('softmax').output

	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
		optimizer=adam)

	return model


#========================================================== Main function ===================================================
mode = 'train'
load_previous = raw_input('Type yes/no if you want to load previous model: ')

print mode

K.set_learning_phase(1)  # all new operations will be in train mode from now on

data_path = train_path
lab_file = train_lab_file

data_gen = DataGenerator(minibatch_size=minibatch_size,
	img_dim=img_dim,
	maxlen=maxlen,
	val_split=val_split,
	nb_classes=nb_classes,
	data_path=data_path,
	lab_file=lab_file)


# Resume training.
if load_previous == 'yes':
	model = load_model()
else:
	model = build_net()


# Early stopping to avoid overfitting.
earlystopping = EarlyStopping(monitor='val_loss',
	patience=20,
	verbose=1)
# Checkpoint to save the weights with the best validation accuracy.
best_model_path = stamp + '.h5'
checkpoint = ModelCheckpoint(best_model_path,
	monitor='val_loss',
	verbose=1,
	save_best_only=True,
	save_weights_only=True,
	mode='auto')

plateau_callback = ReduceLROnPlateau(monitor='loss',
	factor=0.5,
	patience=7,
	min_lr=0.00005,
	verbose=1,
	cooldown=2)

print 'Start training.'
start_time = time.time()

model.fit_generator(generator=data_gen.next_train(),
	steps_per_epoch=(data_gen.get_size(train=True)/minibatch_size),
	epochs=nb_epoch,
	validation_data=data_gen.next_val(),
	validation_steps=(data_gen.get_size(train=False)/minibatch_size),
	callbacks=[earlystopping, checkpoint, data_gen, plateau_callback])

end_time = time.time()
print "--- Training time: %s seconds ---" % (end_time - start_time)