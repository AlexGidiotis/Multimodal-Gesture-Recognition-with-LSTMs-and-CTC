import random
import time
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Input, Lambda, TimeDistributed, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from keras.optimizers import RMSprop, Adam, Nadam
import keras.callbacks
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1,l2
from keras.constraints import maxnorm
from keras import layers
from keras.initializers import RandomUniform

from data_generator import DataGenerator

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
val_split = 0.2
maxlen = 1900
nb_classes = 22
nb_epoch = 500
numfeats_speech = 39
numfeats_skeletal = 20

K.set_learning_phase(1)  # all new operations will be in train mode from now on

# Can load a previous model to resume training. 'yes'
load_previous = raw_input("Load previous model? ")
# Create data generator object.
data_gen = DataGenerator(minibatch_size=minibatch_size,
	numfeats_skeletal=numfeats_skeletal,
	numfeats_speech=numfeats_speech, 
	maxlen=maxlen,
	val_split=val_split,
	nb_classes=nb_classes)

# Load pre-trained uni-modal networks.
skeletal_model_file = '../skeletal_network/sk_ctc_lstm_model.json'
skeletal_weights = '../skeletal_network/sk_ctc_lstm_weights_best.h5'
speech_model_file = '../audio_network/sp_ctc_lstm_model.json'
speech_weights = '../audio_network/sp_ctc_lstm_weights_best.h5'


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
uni_initializer = RandomUniform(minval=-0.05,
	maxval=0.05,
	seed=47)

# Shape of the network inputs.
input_shape_a = (maxlen, numfeats_speech)
input_shape_s = (maxlen, numfeats_skeletal)

# Input layers for the audio and skeletal.
input_data_a = Input(name='the_input_audio',
	shape=input_shape_a,
	dtype='float32')
input_data_s = Input(name='the_input_skeletal',
	shape=input_shape_s,
	dtype='float32')

# Add zero-centered gaussian noise to the input. (For better regularization)
input_noise_a = GaussianNoise(stddev=0.5,
	name='gaussian_noise_a')(input_data_a)
input_noise_s = GaussianNoise(stddev=0.5,
	name='gaussian_noise_s')(input_data_s)


# Blocks 1(audio) run in parallel with 2(skeletal).
# Audio BLSTM
# kernel_constraint=maxnorm(10),
# Block 1
# Load the pre-trained layers and add them to the model.
blstm_1_a = speech_model.layers[2](input_noise_a)
blstm_2_a = speech_model.layers[3](blstm_1_a)
# Build residual connection.
res_a_1 = layers.add([blstm_1_a, blstm_2_a],
	name='speech_residual')
#dropout_a = speech_model.layers[5](res_a_1)

# Skeletal BLSTM
# Block 2
# Load the pre-trained layers and add them to the model.
blstm_1_s = skeletal_model.layers[2](input_noise_s)
blstm_2_s = skeletal_model.layers[3](blstm_1_s)
# Build residual connection.
res_s_1 = layers.add([blstm_1_s, blstm_2_s],
	name='skeletal_residual')
#dropout_s = skeletal_model.layers[5](res_s_1)


# Build the audio model and rename layers.
model_a = Model(input=[input_data_a],
	output=res_a_1)
model_a.layers[2].name='speech_blstm_1'
model_a.layers[3].name='speech_blstm_2'
#model_a.layers[5].name='speech_dropout'

# Build the skeletal model and rename layers.
model_s = Model(input=[input_data_s],
	output=res_s_1)
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
			layer_trainable(l_a,
				freeze=True,
				verbose=True)

	if len(l_s.trainable_weights):
		if type(l_s) in frozen_types:
			layer_trainable(l_s,
				freeze=True,
				verbose=True)


model_a.summary()
model_s.summary()


# Merge the parallel single modality networks.
merged = Merge([model_a, model_s],
	mode='concat')([res_a_1,res_s_1])

# Block 3
lstm_3 = Bidirectional(LSTM(100,
	name='blstm_2',
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	recurrent_dropout=0.0,
	dropout=0.5,
	kernel_constraint=maxnorm(3),
	kernel_initializer=uni_initializer,
	return_sequences=True),
	merge_mode='concat')(merged)


dropout_3 = Dropout(0.6,
	name='dropout_layer_3')(lstm_3)

# Softmax output layer
# Block 4
inner = Dense(nb_classes,
	name='dense_1',
	kernel_initializer=uni_initializer)(dropout_3)
y_pred = Activation('softmax',
	name='softmax')(inner)

Model(input=[input_data_a,input_data_s],
	output=y_pred).summary()


#===================================================== LABELS ==========================================================================================
# These are also inputes needed for the CTC loss
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
loss_out = Lambda(ctc_lambda_func,
	output_shape=(1,),
	name="ctc")([y_pred, labels, input_length, label_length])

# The complete model with the CTC loss.
model = Model(input=[input_data_a,input_data_s, labels, input_length, label_length],
	output=[loss_out])

# Optimizer.
adam = Adam(lr=0.0001,
	clipvalue=0.5,
	decay=1e-5)


# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
	optimizer=adam)

# Early stopping to avoid overfitting
earlystopping = EarlyStopping(monitor='val_loss',
	patience=20,
	verbose=1)
# Checkpoint to save the weights with the best validation accuracy.
filepath="multimodal_ctc_lstm_weights_best.h5"
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
