
import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import groupby
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import RMSprop,Adam
import keras.callbacks
from keras.layers import Input, Lambda
from keras.models import Model
import itertools
from sklearn import preprocessing

from data_generator import DataGenerator


#================================================================== DECODE ==========================================================================
# Gets a batch of predictions and decodes it into predicted sequence.
# THe decoding here is best path with threshold.
def decode_batch(pred_out,f_list):
	# Map gesture codes to classes.
	map_gest = {0:"oov", 1:"Vattene", 2:"Vieni", 3:"qui", 4:"Perfetto", 5:"E'", 6:"un", 7:"furbo", 8:"Che", 9:"due", 10:"palle",
			11:"vuoi", 12:"Vanno", 13:"d'accordo", 14:"Sei", 15:"Pazzo", 16:"Cos'hai", 17:"combinato", 18:"Non", 19:"me", 20:"ne", 21:"frega",
			22:"niente", 23:"ok", 24:"Cosa", 25:"ti", 26:"farei", 27:"Basta", 28:"Le", 29:"prendere", 30:"ce", 31:"n'e", 32:"piu",
			33:"Ho", 34:"fame", 35:"Tanto", 36:"tempo", 37:"fa", 38:"Buonissimo", 39:"Si", 40:"sono", 41:"messi", 42:"stufo" , 43:"sil", -1:"sil"}

	# These files are problematic during decoding.
	ignore_list = [228,298,299,300,303,304,334,343,373,375]

	# Write the output to .mlf
	of = open('ctc_recout.mlf', 'w')
	of.write('#!MLF!#\n')

	out = pred_out
	ret = []
	for j in range(out.shape[0]):
		out_prob = list(np.max(out[j, 2:],1))
		out_best = list(np.argmax(out[j, 2:],1))
		# Filter the probabilities to get the most confident predictions.
		
		for p,s in zip(out_prob,out_best):
			if p < 0.75:
				out_prob.remove(p)
				out_best.remove(s)

		out_best = [k for k, g in itertools.groupby(out_best)]

		outstr = [map_gest[i] for i in out_best]
		ret.append(outstr)

		f_num = f_list[j]

		if int(f_num) in ignore_list:
			continue

		fileNum = str(format(f_num, '05'))
		fileName = 'Sample' + fileNum + '_audio'
		of.write('"*/%s.rec"\n' %fileName)
		for cl in outstr:
			of.write('%s\n' %cl)
		of.write('.\n') 

	of.close()

	return ret

# ============================================================== MAIN ========================================================================

minibatch_size = 2
maxlen = 1900
nb_classes = 44
nb_epoch = 100
numfeats = 39

K.set_learning_phase(0)  # all new operations will be in test mode from now on

dataset = raw_input('select train or val: ')
data_gen = DataGenerator(minibatch_size=minibatch_size, numfeats=numfeats, maxlen=maxlen, nb_classes=nb_classes, dataset=dataset)

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

pred_model = Model(inputs=model.input,
					outputs=model.get_layer('softmax').output)

pred_model.summary()

predictions = pred_model.predict_generator(generator=data_gen.next_val(),
	steps=data_gen.get_size(train=False)/minibatch_size,
	verbose=1)

f_list = data_gen.get_file_list(train=False)

decoded_res = decode_batch(predictions, f_list)
