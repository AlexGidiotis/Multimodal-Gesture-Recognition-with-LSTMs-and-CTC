
import pandas as pd
import numpy as np
import os
import re

from velocity import calculate_hand_velocities
from load_skeleton import import_data
from r_position import estimate_rest_position, calc_distance_from_rp


###################### Main function #############################################################
# Here the inputs are .csv files (one for each video) with the positions of the joints as extracted by kinect
sk_path = "/home/alex/Documents/Data/skeletal_data_csv"
out_path = "/home/alex/Documents/Data/skeletal_activity_feats_csv"


sk_data_path = sk_path
out_path = out_path

sk_data_list = sorted(os.listdir(sk_data_path))

file_count = 0
print "Loading data..."
for data_file in sk_data_list:
	file_count += 1
	if data_file[-4:] != '.csv': continue
	print data_file
	df = import_data(sk_data_path, data_file)
	print "Finished loading data."
	print "Calculating hand velocities..."
	df = calculate_hand_velocities(df)
	print "Estimating rest position..."
	# another error here
	try:
		df, rest_position = estimate_rest_position(df)	
	except:
		continue
	print rest_position
	print "Calculating hand distances from rp..."
	df = calc_distance_from_rp(df,rest_position)
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	print "Writing output to csv..."
	df.to_csv(out_path + '/' + data_file[:-4] + '.csv', index=False)



		




