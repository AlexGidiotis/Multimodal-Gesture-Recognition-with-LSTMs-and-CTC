import pandas as pd
from scipy.spatial.distance import pdist

# calculates the velocity of both hands as the distance in pixels each hand has moved between consecutive frames
# args: df: dataframe with the joint positions
# returns: df: with two more columns appended for left and right hand velocities
def calculate_hand_velocities(df):
	lh_velocities = []
	rh_velocities = []
	for i in df.index:
		# the first frames are usually zero so we do not calculate velocities for them
		if i < 4: 
			lh_velocities.append(0)
			rh_velocities.append(0)
		else:
			cur_lhx, cur_lhy = df.loc[i,'lhX'], df.loc[i,'lhY']
			prev_lhx, prev_lhy = df.loc[(i-1),'lhX'], df.loc[(i-1),'lhY']
			cur_rhx, cur_rhy = df.loc[i,'rhX'], df.loc[i,'rhY']
			prev_rhx, prev_rhy = df.loc[(i-1),'rhX'], df.loc[(i-1),'rhY']
		# euclidean distance between consecutive frames
			lh_velocities.append(int(pdist([[prev_lhx,prev_lhy],[cur_lhx,cur_lhy]], 'euclidean')))
			rh_velocities.append(int(pdist([[prev_rhx,prev_rhy],[cur_rhx,cur_rhy]], 'euclidean')))
	# add the columns to df
	df['lh_v'] = lh_velocities
	df['rh_v'] = rh_velocities

	return df