import os
import random

import pandas as pd
 

audio_train_data = '../audio_network/Training_set_audio.csv'
skeletal_train_data = '../skeletal_network/Training_set_skeletal.csv'
audio_val_data = '../audio_network/Testing_set_audio.csv'
skeletal_val_data = '../skeletal_network/Validation_set_skeletal.csv'
train_labs = '../training.csv'
val_labs = '../validation.csv'


def sample_validation_set():
	"""
	Sample the validation set to get a list of 95 files. These files will be 
	merged with the training set.
	"""

	val_file_list = pd.read_csv(audio_val_data,
		usecols=['file_number'])['file_number'].unique().tolist()

	random.seed(10)
	sample_files = sorted(random.sample(xrange(len(val_file_list)), 95))
	train_file_sample = [val_file_list[i] for i in sample_files]
	val_file_sample = sorted(set(val_file_list) - set(train_file_sample))

	return train_file_sample,val_file_sample


def mix_skeletal_datasets(train_file_sample,
	val_data,
	train_data):
	"""
	Mix the skeletal data of the training set with the data sampled from the
	validation set. The result is two new (training and validation) sets.
	"""

	val_df = pd.read_csv(val_data)


	train_val_df = val_df.loc[val_df['file_number'].isin(train_file_sample)]
	val_df = val_df.loc[~val_df['file_number'].isin(train_file_sample)]


	train_df = pd.read_csv(train_data)


	train_df = pd.concat([train_df,train_val_df],
		ignore_index=True)

	return train_df, val_df


def write_sample_data(data_file,
	file_list,
	dataset):
	"""
	Write the audio data from a of a list of files into separate
	csv files. Creates one csv per audio file.
	"""

	df = pd.read_csv(data_file)
	df = df.loc[df['file_number'].isin(file_list)]

	print dataset
	out_dir = '../data/' + dataset + '_audio'

	for i,file_id in enumerate(file_list):
		if i % 10 == 0:
			print i

		vf = df[df['file_number'] == file_id]

		out_file_name = 'audio_' + str(file_id) + '.csv'
		out_file = os.path.join(out_dir,out_file_name)

		vf.to_csv(out_file,index=False)

	return


def mix_labels(train_file_list,
	train_file_sample):
	"""
	Mix the labels of the training set with the labels sampled from the
	validation set.
	"""

	train_lab_df = pd.read_csv(train_labs)

	val_lab_df = pd.read_csv(val_labs)

	train_val_lab_df = val_lab_df.loc[val_lab_df['Id'].isin(train_file_sample)]
	val_lab_df = val_lab_df.loc[~val_lab_df['Id'].isin(train_file_sample)]

	train_lab_df = pd.concat([train_lab_df,train_val_lab_df],
		ignore_index=True)

	return train_lab_df, val_lab_df



if __name__ == '__main__':
	"""
	This routine expands the training set with a samle from the validation set.
	This is consistently done for both the audio, skeletal and label data.
	"""
	train_file_sample, val_file_sample = sample_validation_set()
	print len(train_file_sample), len(val_file_sample)
	print train_file_sample


	train_file_list = pd.read_csv(audio_train_data,
		usecols=['file_number'])['file_number'].unique().tolist()
	print len(train_file_list)


	print 'Processing label data...'
	train_lab_df,val_lab_df = mix_labels(train_file_list,
		train_file_sample)

	print train_lab_df.shape,val_lab_df.shape

	train_lab_df.to_csv('../data/training.csv',index=False)
	val_lab_df.to_csv('../data/validation.csv',index=False)


	print 'Processing audio data...'
	output_list = [(audio_train_data,train_file_list,'train'),
		(audio_val_data,train_file_sample,'train'),
		(audio_val_data,val_file_sample,'val')]

	for (data_file,file_list,dataset) in output_list:

		print data_file,dataset

		write_sample_data(data_file,file_list,dataset)


	print 'Processing skeletal data...'
	train_skeletal_df,val_skeletal_df = mix_skeletal_datasets(train_file_sample,
		skeletal_val_data,
		skeletal_train_data)

	print train_skeletal_df.shape,val_skeletal_df.shape

	train_skeletal_df.to_csv('../data/Training_set_skeletal.csv',index=False)
	val_skeletal_df.to_csv('../data/Validation_set_skeletal.csv',index=False)
