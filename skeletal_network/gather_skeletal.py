
import pandas as pd
import numpy as np
import re
import os
import time

# Just gathers all the files into one big dataframe.

def load_data(data_path):
	data_listing = sorted(os.listdir(data_path))
	df = pd.DataFrame()
	# go through all files and load all data in a big dataframe
	for c,dfile in enumerate(data_listing):
		new_df = pd.read_csv(data_path + '/' + dfile)
		#Sample00198_data.csv
		file_number = re.findall("Sample(\d*)_data.csv",dfile)[0]
		print file_number
		# this is the training set
		if int(file_number) > 403:
			break
		file_name = pd.Series(file_number, index=new_df.index)
		new_df['file_number'] = file_name
		df = df.append(new_df)
	train_df = df


	df = pd.DataFrame()
	# go through all files and load all data in a big dataframe
	for c,dfile in enumerate(data_listing):
		new_df = pd.read_csv(data_path + '/' + dfile)
		#Sample00198_data.csv
		file_number = re.findall("Sample(\d*)_data.csv",dfile)[0]
		print file_number
		# this is the training set
		if int(file_number) <= 403:
			continue
		file_name = pd.Series(file_number, index=new_df.index)
		new_df['file_number'] = file_name
		df = df.append(new_df)
	val_df = df

	return train_df,val_df


data_path = "/home/alex/Documents/Data/skeletal_activity_feats_csv"
print "Loading data set..."
train_df,val_df = load_data(data_path)

train_df.to_csv("training_data.csv",
	index=False)

val_df.to_csv("validation_data.csv",
	index=False)