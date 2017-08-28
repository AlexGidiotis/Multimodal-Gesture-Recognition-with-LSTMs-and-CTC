
import pandas as pd
import numpy as np
import re
import os
import time

# Just gathers all the files into one big dataframe.

def load_data(data_path):
	data_listing = sorted(os.listdir(data_path))
	df = pd.DataFrame()
	count = 0
	# go through all files and load all data in a big dataframe
	for dfile in data_listing:
		new_df = pd.read_csv(data_path + '/' + dfile)
		#Sample00198_data.csv
		file_number = re.findall("Sample(\d*)_SKData.csv",dfile)[0]
		file_name = pd.Series(file_number, index=new_df.index)
		new_df['file_number'] = file_name
		df = df.append(new_df)
	return df


data_path = "/home/alex/Documents/Data/CSV_TRAIN_data"
print "Loading data set..."
df = load_data(data_path)

print df

df.to_csv("training_data.csv", index=False)