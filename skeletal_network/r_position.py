import pandas as pd
from scipy.spatial.distance import pdist

# estimate the rest position as the median position of all segments with low velocity
# args: df: dataframe with skeleton joints
# returns: df: dataframe with the 'low_velocity' column added
#		   rp: a tuple of 16 integers with the rest position of both hands (lsx,lsy,lex,ley,lwx,lwy,lhx,lhy,rsx,rsy,rex,rey,rwx,rwy,rhx,rhy)
def estimate_rest_position(df):
	median_left = df['lh_v'].mean()
	median_right = df['rh_v'].mean()

	#flag as low velocity all frames that have both hands under velocity threshold
	df['low_velocity'] = (df['lh_v']<median_left)&(df['rh_v']<median_right)
	low_v = df[df['low_velocity'] == True]

	# estimate the rest position
	rp = int(low_v['lsX'].median()),int(low_v['lsY'].median()),int(low_v['leX'].median()),int(low_v['leY'].median()),\
		int(low_v['lwX'].median()),int(low_v['lwY'].median()),int(low_v['lhX'].median()), int(low_v['lhY'].median()),\
		int(low_v['rsX'].median()), int(low_v['rsY'].median()),int(low_v['reX'].median()), int(low_v['reY'].median()),\
		int(low_v['rwX'].median()), int(low_v['rwY'].median()),int(low_v['rhX'].median()), int(low_v['rhY'].median())
	return df, rp

# calculates the euclidean distance of both hands from the rest position
# args: df: dataframe witht the skeleton joints
#		rp: a tuple of 16 integers with the rest position for both hands joints
# returns: df: dataframe with 'lh_dist_rp' and 'rh_dist_rp' columns added to it
def calc_distance_from_rp(df,rp):
	# unpack rp values
	rp_lsx,rp_lsy,rp_lex,rp_ley,rp_lwx,rp_lwy,rp_lhx,rp_lhy,rp_rsx,rp_rsy,rp_rex,rp_rey,rp_rwx,rp_rwy,rp_rhx,rp_rhy = rp
	lh_dist_rp = []
	rh_dist_rp = []
	# the first frames are usually zero so we do not calculate distances for them
	for i in df.index:
		if i < 4: 
			lh_dist_rp.append(0)
			rh_dist_rp.append(0) 
		else:	
			cur_lhx, cur_lhy = df.loc[i,'lhX'], df.loc[i,'lhY']
			cur_rhx, cur_rhy = df.loc[i,'rhX'], df.loc[i,'rhY']
		# euclidean distance between hand position and rp
			lh_dist_rp.append(int(pdist([[rp_lhx, rp_lhy],[cur_lhx,cur_lhy]], 'euclidean')))
			rh_dist_rp.append(int(pdist([[rp_rhx, rp_rhy],[cur_rhx,cur_rhy]], 'euclidean')))
	# add the columns to df
	df['lh_dist_rp'] = lh_dist_rp
	df['rh_dist_rp'] = rh_dist_rp
	return df
