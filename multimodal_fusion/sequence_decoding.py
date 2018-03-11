
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

from data_generator import DataGenerator
from losses import ctc_lambda_func


def decode_batch(pred_out,f_list):
	"""
	"""

	# Map gesture codes to classes.
	map_gest = {0:"oov", 1:"VA", 2:"VQ", 3:"PF", 4:"FU", 5:"CP", 6:"CV",
				7:"DC", 8:"SP", 9:"CN", 10:"FN", 11:"OK", 12:"CF", 13:"BS",
				14:"PR", 15:"NU", 16:"FM", 17:"TT",  18:"BN",  19:"MC",
				20:"ST", 21:"sil"}

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
			if p < 0.8:
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


if __name__ == '__main__':

	minibatch_size = 2
	maxlen = 1900
	nb_classes = 22
	nb_epoch = 100
	numfeats_speech = 39
	numfeats_skeletal = 20

	K.set_learning_phase(0)  # all new operations will be in test mode from now on

	dataset = raw_input('select train or val: ')

	data_gen = DataGenerator(minibatch_size=minibatch_size,
		numfeats_speech=numfeats_speech,
		numfeats_skeletal=numfeats_skeletal,
		maxlen=maxlen,
		nb_classes=nb_classes,
		dataset=dataset)

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

	predictions = pred_model.predict_generator(generator=data_gen.next_val(),
		steps=data_gen.get_size(train=False)/minibatch_size,
		verbose=1)

	f_list = data_gen.get_file_list(train=False)

	decoded_res = decode_batch(predictions, f_list)
