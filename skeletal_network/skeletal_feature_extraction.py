# Author: Alex Gidiotis
#		  gidiotisAlex@outlook.com.gr
#
# Goes through the skeletal feature extraction process.
# Reads from the skeletal data and label files.
# Outputs a .csv file with the whole dataset features extracted.

import re
import os
import numpy as np
import pandas as pd
from operator import itemgetter
from itertools import groupby

# ========================================= Loads the skeletal data and labels =================================================================
# Returns: a dataframe with the whole training set frame by frame. (num frames x 28)
def load_data(sk_data_file):
	# Go find all the saved data.
	df = pd.read_csv(sk_data_file)
	return df

# ================================= Get the previous position of both hands and elbows. ========================================================
# We are going to use this positions to calculate movements and velocities.
# Returns: the original df with the hands and elbows previous positions added.
def get_previous_pos(df):
	# Put the indices into matrices for faster manipulation.
	lh_x,lh_y,rh_x,rh_y = df['lhX'].as_matrix(),df['lhY'].as_matrix(),df['rhX'].as_matrix(),df['rhY'].as_matrix()
	le_x,le_y,re_x,re_y = df['leX'].as_matrix(),df['leY'].as_matrix(),df['reX'].as_matrix(),df['reY'].as_matrix()

	# Create zero-like matrices.
	pr_lh_x,pr_lh_y,pr_rh_x,pr_rh_y = np.zeros_like(lh_x),np.zeros_like(lh_y),np.zeros_like(rh_x),np.zeros_like(rh_y)
	pr_le_x,pr_le_y,pr_re_x,pr_re_y = np.zeros_like(le_x),np.zeros_like(le_y),np.zeros_like(re_x),np.zeros_like(re_y)

	# Get the previous positions.
	pr_lh_x[1:],pr_lh_y[1:],pr_rh_x[1:],pr_rh_y[1:] = lh_x[:-1],lh_y[:-1],rh_x[:-1],rh_y[:-1]
	pr_le_x[1:],pr_le_y[1:],pr_re_x[1:],pr_re_y[1:] = le_x[:-1],le_y[:-1],re_x[:-1],re_y[:-1]

	# Put them in the dataframes.
	df['pre_lhX'],df['pre_lhY'],df['pre_rhX'],df['pre_rhY'] = pr_lh_x,pr_lh_y,pr_rh_x,pr_rh_y
	df['pre_leX'],df['pre_leY'],df['pre_reX'],df['pre_reY'] = pr_le_x,pr_le_y,pr_re_x,pr_re_y

	return df

#================================= Get the previous velocity of both hands and elbows ==========================================================
# THis will be used to calculate accelerations
def get_previous_vel(df):
	# Put the velocities into matrices for faster manipulation.
	lh_vel, rh_vel = df['lh_v'].as_matrix(),df['rh_v'].as_matrix()
	le_vel, re_vel = df['le_v'].as_matrix(),df['re_v'].as_matrix()

	# Create zero-like matrices.
	pr_lh_vel,pr_rh_vel = np.zeros_like(lh_vel),np.zeros_like(rh_vel)
	pr_le_vel,pr_re_vel = np.zeros_like(le_vel),np.zeros_like(re_vel)

	# Get the previous positions.
	pr_lh_vel[1:],pr_rh_vel[1:] = lh_vel[:-1],rh_vel[:-1]
	pr_le_vel[1:],pr_re_vel[1:] = le_vel[:-1],re_vel[:-1]

	# Put them in the dataframes.
	df['pre_lh_v'],df['pre_rh_v'] = pr_lh_vel,pr_rh_vel
	df['pre_le_v'],df['pre_re_v'] = pr_le_vel,pr_re_vel

	return df


# ======================================== Calculate and return hand and elbow velocities. =====================================================
# We use the current and previous positions of elbows and hands to calculate the velocities as the distance in pixels between consecutive frames.
# Returns the original data frame with the velocity columns added.
def calculate_velocities(df):
	# Load positions to arrays for faster computation.
	lh_x,lh_y,rh_x,rh_y = df['lhX'].as_matrix(),df['lhY'].as_matrix(),df['rhX'].as_matrix(),df['rhY'].as_matrix()
	le_x,le_y,re_x,re_y = df['leX'].as_matrix(),df['leY'].as_matrix(),df['reX'].as_matrix(),df['reY'].as_matrix()
	
	pr_lh_x,pr_lh_y,pr_rh_x,pr_rh_y = df['pre_lhX'].as_matrix(),df['pre_lhY'].as_matrix(),df['pre_rhX'].as_matrix(),df['pre_rhY'].as_matrix()
	pr_le_x,pr_le_y,pr_re_x,pr_re_y = df['pre_leX'].as_matrix(),df['pre_leY'].as_matrix(),df['pre_reX'].as_matrix(),df['pre_reY'].as_matrix()

	# Create the zero arrays to store velocities.
	lh_vel,rh_vel = np.zeros_like(lh_x), np.zeros_like(rh_x)
	le_vel,re_vel = np.zeros_like(le_x), np.zeros_like(re_x)

	# Create the position vectors from the x,y vectors.
	lh, pre_lh, rh, pre_rh = np.array((lh_x,lh_y)), np.array((pr_lh_x,pr_lh_y)), np.array((rh_x,rh_y)), np.array((pr_rh_x,pr_rh_y))
	le, pre_le, re, pre_re = np.array((le_x,le_y)), np.array((pr_le_x,pr_le_y)), np.array((re_x,re_y)), np.array((pr_re_x,pr_re_y))

	# Vectorized computation of the euclidean distance between the previous and current position.
	dist_lh, dist_rh = (lh - pre_lh)**2, (rh - pre_rh)**2
	dist_le, dist_re = (le - pre_le)**2, (re - pre_re)**2

	dist_lh, dist_rh = dist_lh.sum(axis=0), dist_rh.sum(axis=0) 
	dist_le, dist_re = dist_le.sum(axis=0), dist_re.sum(axis=0)

	dist_lh, dist_rh = np.sqrt(dist_lh), np.sqrt(dist_rh) 
	dist_le, dist_re = np.sqrt(dist_le), np.sqrt(dist_re)

	# Store the velocities back to the dataframe.
	lh_vel[5:],rh_vel[5:] = dist_lh[5:], dist_rh[5:]
	le_vel[5:],re_vel[5:] = dist_le[5:], dist_re[5:]

	df['lh_v'], df['rh_v'] = lh_vel, rh_vel	
	df['le_v'], df['re_v'] = le_vel, re_vel
	return df

#================================= Get the acceleration of both hands and elbows ==========================================================
# We use the current and previous velocity of the hands and elbows to calculate acceleration
def calculate_accelerations(df):
	# Load positions to arrays for faster computation.
	lh_vel,rh_vel = df['lh_v'].as_matrix(),df['rh_v'].as_matrix()
	le_vel,re_vel = df['le_v'].as_matrix(),df['re_v'].as_matrix()
	
	pr_lh_vel,pr_rh_vel = df['pre_lh_v'].as_matrix(),df['pre_rh_v'].as_matrix()
	pr_le_vel,pr_re_vel = df['pre_le_v'].as_matrix(),df['pre_re_v'].as_matrix()

	# Create the zero arrays to store velocities.
	lh_acc,rh_acc = np.zeros_like(lh_vel), np.zeros_like(rh_vel)
	le_acc,re_acc = np.zeros_like(le_vel), np.zeros_like(re_vel)

	lh_dv, rh_dv = (lh_vel - pr_lh_vel), (rh_vel - pr_rh_vel)
	le_dv, re_dv = (le_vel - pr_le_vel), (re_vel - pr_re_vel)

	# Store the accelerations back to the dataframe.
	lh_acc[5:],rh_acc[5:] = lh_dv[5:], rh_dv[5:]
	le_acc[5:],re_acc[5:] = le_dv[5:], re_dv[5:]

	df['lh_a'], df['rh_a'] = lh_acc, rh_acc	
	df['le_a'], df['re_a'] = le_acc, re_acc
	return df

# ============================== Calculate the distances of joints from the hip center and shoulder center =====================================
# Calculates the distances of both elbows and hands from the hip center and the shoulder center.
# Returns: the original dataframe with the hip-elbows, hip-hands, should center-elbows and shoulder center-hands distances columns added.
def calculate_distances(df):
	# Load the joint positions into arrays for faster computation.
	lh_x,lh_y,rh_x,rh_y = df['lhX'].as_matrix(),df['lhY'].as_matrix(),df['rhX'].as_matrix(),df['rhY'].as_matrix()
	le_x,le_y,re_x,re_y = df['leX'].as_matrix(),df['leY'].as_matrix(),df['reX'].as_matrix(),df['reY'].as_matrix()
	hip_x,hip_y,shc_x,shc_y = df['hipX'].as_matrix(),df['hipY'].as_matrix(),df['shcX'].as_matrix(),df['shcY'].as_matrix()

	# Create the empty arrays to put the values.
	hands_dist = np.zeros_like(lh_x)
	lh_hip_dist,rh_hip_dist = np.zeros_like(lh_x), np.zeros_like(rh_x)
	le_hip_dist,re_hip_dist = np.zeros_like(le_x), np.zeros_like(re_x)
	lh_shoulder_center_dist,rh_shoulder_center_dist = np.zeros_like(lh_x), np.zeros_like(rh_x)
	le_shoulder_center_dist,re_shoulder_center_dist = np.zeros_like(le_x), np.zeros_like(re_x)

	# Create the position vectors from the x,y vectors.
	lh, rh = np.array((lh_x,lh_y)), np.array((rh_x,rh_y))
	le, re = np.array((le_x,le_y)), np.array((re_x,re_y))
	hip, shc = np.array((hip_x,hip_y)), np.array((shc_x,shc_y))

	# Calculate the euclidean distance between hands.
	hands_dist = (lh - rh)**2
	hands_dist = hands_dist.sum(axis=0)
	hands_dist = np.sqrt(hands_dist)

	# Calculate the euclidean distance of hands and elbows from the hip.
	lh_hip_dist, rh_hip_dist = (lh - hip)**2, (rh - hip)**2
	le_hip_dist, re_hip_dist = (le - hip)**2, (re - hip)**2

	lh_hip_dist, rh_hip_dist = lh_hip_dist.sum(axis=0), rh_hip_dist.sum(axis=0) 
	le_hip_dist, re_hip_dist = le_hip_dist.sum(axis=0), re_hip_dist.sum(axis=0)

	lh_hip_dist, rh_hip_dist = np.sqrt(lh_hip_dist), np.sqrt(rh_hip_dist) 
	le_hip_dist, re_hip_dist = np.sqrt(le_hip_dist), np.sqrt(re_hip_dist)

	# Calculate the euclidean distance of hands and elbows from the shoulder center.
	lh_shoulder_center_dist, rh_shoulder_center_dist = (lh - shc)**2, (rh - shc)**2
	le_shoulder_center_dist, re_shoulder_center_dist = (le - shc)**2, (re - shc)**2

	lh_shoulder_center_dist, rh_shoulder_center_dist = lh_shoulder_center_dist.sum(axis=0), rh_shoulder_center_dist.sum(axis=0) 
	le_shoulder_center_dist, re_shoulder_center_dist = le_shoulder_center_dist.sum(axis=0), re_shoulder_center_dist.sum(axis=0)

	lh_shoulder_center_dist, rh_shoulder_center_dist = np.sqrt(lh_shoulder_center_dist), np.sqrt(rh_shoulder_center_dist) 
	le_shoulder_center_dist, re_shoulder_center_dist = np.sqrt(le_shoulder_center_dist), np.sqrt(re_shoulder_center_dist)

	# Put the distances back into the dataframe to be returned.
	df['hands_d'] = hands_dist

	df['lh_hip_d'], df['rh_hip_d'] = lh_hip_dist, rh_hip_dist	
	df['le_hip_d'], df['re_hip_d'] = le_hip_dist, re_hip_dist

	df['lh_shc_d'], df['rh_shc_d'] = lh_shoulder_center_dist, rh_shoulder_center_dist	
	df['le_shc_d'], df['re_shc_d'] = le_shoulder_center_dist, re_shoulder_center_dist

	return df 

#================================================== Calculate some sets of angles ==============================================================
# Here we calculate the angles of hands-hip, hands-shoulder center and hands-elbows. 
# We represent this angles with 8-chain codes where: 0 is movement right, 1 is up-right, 2 is up, 3 is up-left, 4 is left, 5 is down-left, six is down, 7 is down-right.
# Returns: the original df with the angles columns added to it.
def calculate_angles(df):

	# Load all the positions into arrays for faster computation.
	lh_x,lh_y,rh_x,rh_y = df['lhX'].as_matrix(),df['lhY'].as_matrix(),df['rhX'].as_matrix(),df['rhY'].as_matrix()
	le_x,le_y,re_x,re_y = df['leX'].as_matrix(),df['leY'].as_matrix(),df['reX'].as_matrix(),df['reY'].as_matrix()
	hip_x,hip_y,shc_x,shc_y = df['hipX'].as_matrix(),df['hipY'].as_matrix(),df['shcX'].as_matrix(),df['shcY'].as_matrix()

	# Create the position vectors from the x,y vectors.
	lh, rh = np.array((lh_x,lh_y)), np.array((rh_x,rh_y))
	le, re = np.array((le_x,le_y)), np.array((re_x,re_y))
	hip, shc = np.array((hip_x,hip_y)), np.array((shc_x,shc_y))

	# The number of chain codes we are going to use as described.
	num_chain = 8

	# Calculate the distances between joints.
	lh_hip_dist, rh_hip_dist = np.array((lh-hip)), np.array((rh-hip))
	lh_shc_dist, rh_shc_dist = np.array((lh-shc)), np.array((rh-shc))
	lh_el_dist, rh_el_dist = np.array((lh-le)), np.array((rh-re))

	# Calculate some angles.
	Theta_lh_hip, Theta_rh_hip = np.arctan2(lh_hip_dist[1],lh_hip_dist[0]), np.arctan2(rh_hip_dist[1],rh_hip_dist[0])
	Theta_lh_shc, Theta_rh_shc = np.arctan2(lh_shc_dist[1],lh_shc_dist[0]), np.arctan2(rh_shc_dist[1],rh_shc_dist[0])
	Theta_lh_el, Theta_rh_el = np.arctan2(lh_el_dist[1],lh_el_dist[0]), np.arctan2(rh_el_dist[1],rh_el_dist[0])

	# Calculate the relative angles and convert to chain codes.
	Theta_lh_hip_t = (num_chain - (Theta_lh_hip/(np.pi/4) + 0.5 + (lh_hip_dist[1]<0)*num_chain).astype(int))%num_chain
	Theta_rh_hip_t = (num_chain - (Theta_rh_hip/(np.pi/4) + 0.5 + (rh_hip_dist[1]<0)*num_chain).astype(int))%num_chain
	Theta_lh_shc_t = (num_chain - (Theta_lh_shc/(np.pi/4) + 0.5 + (lh_shc_dist[1]<0)*num_chain).astype(int))%num_chain
	Theta_rh_shc_t = (num_chain - (Theta_rh_shc/(np.pi/4) + 0.5 + (rh_shc_dist[1]<0)*num_chain).astype(int))%num_chain
	Theta_lh_el_t = (num_chain - (Theta_lh_el/(np.pi/4) + 0.5 + (lh_el_dist[1]<0)*num_chain).astype(int))%num_chain
	Theta_rh_el_t = (num_chain - (Theta_rh_el/(np.pi/4) + 0.5 + (rh_el_dist[1]<0)*num_chain).astype(int))%num_chain

	# Store back into the data frame to be returned.
	df['lh_hip_ang'], df['rh_hip_ang'] = Theta_lh_hip_t, Theta_rh_hip_t	
	df['lh_shc_ang'], df['rh_shc_ang'] = Theta_lh_shc_t, Theta_rh_shc_t
	df['lh_el_ang'], df['rh_el_ang'] = Theta_lh_el_t, Theta_rh_el_t	

	return df


# ======================================= Calculate the movement directions of hands. ==========================================================
# Get the movement direction of the hands from the relative to the previous position of each hand. 
# Again these directions are represented using 8-chain codes.
# Returns: the data frame with the left and right hand direction columns added.
def calculate_movement_directions(df):
	# Load current and previous indices into arrays for fast computation.
	lh_x,lh_y,rh_x,rh_y = df['lhX'].as_matrix(),df['lhY'].as_matrix(),df['rhX'].as_matrix(),df['rhY'].as_matrix()
	pr_lh_x,pr_lh_y,pr_rh_x,pr_rh_y = df['pre_lhX'].as_matrix(),df['pre_lhY'].as_matrix(),df['pre_rhX'].as_matrix(),df['pre_rhY'].as_matrix()

	# Create the zero arrays to store velocities.
	lh_direction,rh_direction = np.zeros_like(lh_x), np.zeros_like(rh_x)

	# Create the position vectors from the x,y vectors.
	lh, pre_lh, rh, pre_rh = np.array((lh_x,lh_y)), np.array((pr_lh_x,pr_lh_y)), np.array((rh_x,rh_y)), np.array((pr_rh_x,pr_rh_y))

	# The number of chain codes we are going to use.
	num_chain = 8

	# Calculate the distances from the previous position.
	lh_movement_dist, rh_movement_dist = np.array((lh-pre_lh)), np.array((rh-pre_rh))

	# Calculate some angles.
	Theta_lh, Theta_rh = np.arctan2(lh_movement_dist[1],lh_movement_dist[0]), np.arctan2(rh_movement_dist[1],rh_movement_dist[0])

	# Calculate the relative to the previous position angle and convert to chain codes.
	Theta_lh_t = (num_chain - (Theta_lh/(np.pi/4) + 0.5 + (lh_movement_dist[1]<0)*num_chain).astype(int))%num_chain
	Theta_rh_t = (num_chain - (Theta_rh/(np.pi/4) + 0.5 + (rh_movement_dist[1]<0)*num_chain).astype(int))%num_chain

	# Save the direction arrays.
	lh_direction[5:],rh_direction[5:] = Theta_lh_t[5:], Theta_rh_t[5:]

	# Store to the data frame to be returned.
	df['lh_dir'], df['rh_dir'] = lh_direction, rh_direction	
	return df

#====================================================== Write labels to training datafile =======================================================
def write_train_labs(train_df):
	# Map gesture names to numbers
	gestures_dict = {'vattene':1, 'vieniqui':2, 'perfetto':3, 'furbo':4, 'cheduepalle':5, 'chevuoi':6, 'daccordo':7, 'seipazzo':8, 'combinato':9,
				 'freganiente':10, 'ok':11, 'cosatifarei':12, 'basta':13, 'prendere':14, 'noncenepiu':15, 'fame':16, 'tantotempo':17, 
				 'buonissimo':18, 'messidaccordo':19, 'sonostufo':20}

	# Here are saved the labeling data
	training_lab_path = '/home/alex/Documents/Data/Labels'
	# save all gestures with end frame, start frame and video number in a df to match the train_df
	lab_frame = pd.DataFrame()
	# Go through all lab files and load the label info in a dataframe
	for lab_file in sorted(os.listdir(training_lab_path)):
		lf = pd.read_csv(training_lab_path + '/' + lab_file, delimiter=' ', header=None).drop([1,3],1)
		# Get file number
		file_number = re.findall("Sample(\d*)_data_labels",lab_file)[0]
		lf['file_number'] = pd.Series(int(file_number), index=lf.index)
		lab_frame = lab_frame.append(lf)

	# Put all labels in one big vector
	label_vector = []
	# Go through each file
	for f_num in train_df['file_number'].unique():
		#Labeling data for the file
		lab_f = lab_frame[lab_frame['file_number'] == f_num]
		# THe dataframe of the file
		tr_f = train_df[train_df['file_number'] == f_num]
		# vector of frames
		frames = tr_f['frame'].as_matrix()
		# vector of activity/inactivity info
		inactive = tr_f['inactive'].as_matrix()
		# label vector to be created
		lab_v = np.zeros_like(inactive)
		
		# use the label and activity/inactivity info to label frames (inactive -> 0)
		for f, i in zip(frames, inactive):
			#if inactive do nothing
			if i == 1: continue
			try:
				gest_lab = lab_f[(lab_f[2] < f) & (lab_f[4] >= f)][0].values[0]
			# if there is no label do nothing
			except: continue
			# map gesture name to number
			lab_v[f] = gestures_dict[gest_lab]
		# append to the big vector
		label_vector = np.hstack((label_vector,lab_v))	

	train_df['labels'] = label_vector.astype(int)

	return train_df

#============================================================= Main function ====================================================================
# This is the function that goes through the feature extraction process and writes the output.
sk_data_file = "training_data.csv"

print "Loading data..."
df = load_data(sk_data_file) 
print "Finished loading."
df = get_previous_pos(df)
print "Calculating velocities..."
df = calculate_velocities(df)
df = get_previous_vel(df)
print "Calculating accelerations..."
df = calculate_accelerations(df)
print "Calculating distances..."
df = calculate_distances(df)
print "Calculating angles..."
df = calculate_angles(df)
print "Calculating movement directions..."
df = calculate_movement_directions(df)
print "Writing output to csv..."
df.to_csv("Training_set_skeletal.csv",index=False)











