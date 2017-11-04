import os
import re
import pandas as pd
import numpy as np
import cv2

#========================================================= Global variables =================================================
train_file_skeletal = '/home/alex/Documents/Python/multimodal_gesture_recognition/Training_set_skeletal.csv'
validation_file_skeletal = '/home/alex/Documents/Python/multimodal_gesture_recognition/Validation_set_skeletal.csv'
train_video_path = '/home/alex/Documents/Data/Color_vid'
validation_video_path = '/home/alex/Documents/Data/Test_Color_vid'
train_out_path = '/home/alex/Documents/Data/training_up_body'
validation_out_path = '/home/alex/Documents/Data/validation_up_body'

img_dim = 60

#=========================================================== Definitions ====================================================
def extract_body(df,video_path,out_path):
	for c, vfile in enumerate(sorted(os.listdir(video_path))):
		# Ignore other files.
		if vfile[-4:] != '.mp4':
			continue
		# Extract the file number from the file name.
		file_num = int(re.findall('Sample(\d*)_',vfile)[0])
		print file_num
		# Get the skeletal data for the particullar video.
		vf = df[df['file_number'] == file_num]
		# THe position of the hip center and the shoulder center.
		hipX, hipY, shcX, shcY = vf['hipX'].tolist(), vf['hipY'].tolist(), vf['shcX'].tolist(), vf['shcY'].tolist()
		lhX, lhY, rhX, rhY = vf['lhX'].tolist(), vf['lhY'].tolist(), vf['rhX'].tolist(), vf['rhY'].tolist()

		X_data = []
		# Create video stream.
		video = os.path.join(video_path,vfile)
		cap = cv2.VideoCapture(video)
		# Frame counter reset.
		frame = 0
		while cap.isOpened():
			# Check for eof.
			ret, img = cap.read()
			if not ret: break
			# rgb to grayscale.
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			try:
				up = shcY[frame] - 120 
				down = hipY[frame] + 120
				left = hipX[frame] - 180
				right = hipX[frame] + 180
				# Check for out of range.
				if up <= 0 : up = 1
				if down >= 480 : down = 479
				if left <= 0 : left = 1
				if right >= 640 : right =639
				# Crop the grayscaled image with the upper body.
				crop_img = gray_img[up:down,left:right]

				# Downsample to 64x64 pixels.
				res_img = cv2.resize(crop_img,(img_dim,img_dim),
					interpolation = cv2.INTER_CUBIC)
				res_img = res_img.reshape([img_dim,img_dim,1])
				X_data.append(res_img)
			# If unable to use the skeletal info do this.
			except:
				crop_img = gray_img[0:330,0:640]

				res_img = cv2.resize(crop_img,(img_dim,img_dim),
					interpolation = cv2.INTER_CUBIC)
				res_img = res_img.reshape([img_dim,img_dim,1])
				X_data.append(res_img)

			# INcrease the frame count.
			frame += 1

		# Save the video ndarray to .npy format
		X_data = np.array(X_data)
		out_file_name = vfile[:-4] + '.npy'
		out_file = os.path.join(out_path,out_file_name)
		np.save(out_file,X_data)

	return 

#========================================================== Main function ===================================================
# Choose between train and test mode. No difference between train and test data, just different paths.
mode = raw_input('Choose train or validation: ')
print mode

if mode == 'train':
	df = pd.read_csv(train_file_skeletal)
	video_path = train_video_path
	out_path = train_out_path
elif mode == 'validation':
	df = pd.read_csv(validation_file_skeletal)
	video_path = validation_video_path
	out_path = validation_out_path

if not os.path.exists(out_path):
	os.makedirs(out_path)
	print 'directory %s created' % out_path

extract_body(df,video_path,out_path)