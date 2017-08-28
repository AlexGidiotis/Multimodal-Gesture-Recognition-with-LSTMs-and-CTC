import pandas as pd
import numpy as np

# The arrays loaded are not in proper format so we modify them to x,y pairs. Also we filter irrelevant values.
def modify_array(arr):
	# arrays to be returned 
	arr_x = []
	arr_y = []
	for item in arr:
		# Get the items in proper format
		item = item.strip('[').strip(']').split()
		# Filter values
		if int(item[0]) >= 640 : item[0] = 320
		if int(item[1]) >= 480 : item[1] = 240
		# Save to the lists to be returned
		arr_x.append(int(item[0]))
		arr_y.append(int(item[1]))

	return arr_x, arr_y

# Opens a data file and imports values
# args:	sk_data_path: path to the data folder
#		data_file: file to be imported
# returns:	df: a dataframe with all the imported values

# skeletal data files come in the following format:	
# Frame: f Hip,Shoulder_Center,Left: lsx,lsy lex,ley lwx,lwy lhx,lhy Right: rsx,rsy rex,rey rwx,rwy rhx,rhy
def import_data(sk_data_path, data_file):
	data_f = open(sk_data_path + '/' + data_file, 'r')

	# Read the data from csv file
	read_df = pd.read_csv(data_f)
	frame = read_df['Unnamed: 0'].as_matrix()
	hip = read_df['hip_center'].as_matrix()
	shoulder_cent = read_df['shoulder_center'].as_matrix()
	l_shoulder = read_df['left_shoulder'].as_matrix()
	l_elbow = read_df['left_elbow'].as_matrix()
	l_wrist = read_df['left_wrist'].as_matrix()
	l_hand = read_df['left_hand'].as_matrix()
	r_shoulder = read_df['right_shoulder'].as_matrix()
	r_elbow = read_df['right_elbow'].as_matrix()
	r_wrist = read_df['right_wrist'].as_matrix()
	r_hand = read_df['right_hand'].as_matrix()

	# Create the new dataframe for further processing
	df = pd.DataFrame()
	df['frame'] = frame
	df['hipX'], df['hipY'] = modify_array(hip)
	df['shcX'], df['shcY'] = modify_array(shoulder_cent)
	df['lsX'], df['lsY'] = modify_array(l_shoulder)
	df['leX'], df['leY'] = modify_array(l_elbow)
	df['lwX'], df['lwY'] = modify_array(l_wrist)
	df['lhX'], df['lhY'] = modify_array(l_hand)
	df['rsX'], df['rsY'] = modify_array(r_shoulder)
	df['reX'], df['reY'] = modify_array(r_elbow)
	df['rwX'], df['rwY'] = modify_array(r_wrist)
	df['rhX'], df['rhY'] = modify_array(r_hand)

	return df