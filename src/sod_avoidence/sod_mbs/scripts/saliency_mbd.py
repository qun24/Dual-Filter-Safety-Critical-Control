import math
import sys
import operator
import time
import networkx as nx
import numpy as np
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.optimize import minimize
import os
import cv2
#import pdb

def raster_scan(img,L,U,D):
	n_rows = len(img)
	n_cols = len(img[0])

	for x in range(1,n_rows - 1):
		for y in range(1,n_cols - 1):
			ix = img[x][y]
			d = D[x][y]

			u1 = U[x-1][y]
			l1 = L[x-1][y]

			u2 = U[x][y-1]
			l2 = L[x][y-1]

			b1 = max(u1,ix) - min(l1,ix)
			b2 = max(u2,ix) - min(l2,ix)

			if d <= b1 and d <= b2:
				continue
			elif b1 < d and b1 <= b2:
				D[x][y] = b1
				U[x][y] = max(u1,ix)
				L[x][y] = min(l1,ix)
			else:
				D[x][y] = b2
				U[x][y] = max(u2,ix)
				L[x][y] = min(l2,ix)

	return True

def raster_scan_inv(img,L,U,D):
	n_rows = len(img)
	n_cols = len(img[0])

	for x in range(n_rows - 2,1,-1):
		for y in range(n_cols - 2,1,-1):

			ix = img[x][y]
			d = D[x][y]

			u1 = U[x+1][y]
			l1 = L[x+1][y]

			u2 = U[x][y+1]
			l2 = L[x][y+1]

			b1 = max(u1,ix) - min(l1,ix)
			b2 = max(u2,ix) - min(l2,ix)

			if d <= b1 and d <= b2:
				continue
			elif b1 < d and b1 <= b2:
				D[x][y] = b1
				U[x][y] = max(u1,ix)
				L[x][y] = min(l1,ix)
			else:
				D[x][y] = b2
				U[x][y] = max(u2,ix)
				L[x][y] = min(l2,ix)

	return True

def mbd(img, num_iters):

	if len(img.shape) != 2:
		print('did not get 2d np array to fast mbd')
		return None
	if (img.shape[0] <= 3 or img.shape[1] <= 3):
		print('image is too small')
		return None

	L = np.copy(img)
	U = np.copy(img)
	D = float('Inf') * np.ones(img.shape)
	D[0,:] = 0
	D[-1,:] = 0
	D[:,0] = 0
	D[:,-1] = 0

	# unfortunately, iterating over numpy arrays is very slow
	img_list = img.tolist()
	L_list = L.tolist()
	U_list = U.tolist()
	D_list = D.tolist()

	for x in range(0,num_iters):
		if x%2 == 1:
			raster_scan(img_list,L_list,U_list,D_list)
		else:
			raster_scan_inv(img_list,L_list,U_list,D_list)

	return np.array(D_list)


def get_saliency_mbd(input,method='b'):

	img_path_list = []
	#we get either a filename or a list of filenames
	if type(input) == type(str()):
		img_path_list.append(input)
	elif type(input) == type(list()):
		img_path_list = input
	else:
		print('Input type is neither list or string')
		return None

	# Saliency map calculation based on: Minimum Barrier Salient Object Detection at 80 FPS
	for img_path in img_path_list:

		img = skimage.io.imread(img_path)
		
		if img.shape[2] == 4: img = img[:, :, :3] 
	
		img_mean = np.mean(img,axis=(2))
		sal = mbd(img_mean,3)


		if method == 'b':
			# get the background map

			# paper uses 30px for an image of size 300px, so we use 10%
			(n_rows,n_cols,n_channels) = img.shape
			img_size = math.sqrt(n_rows * n_cols)
			border_thickness = int(math.floor(0.1 * img_size))

			img_lab = img_as_float(skimage.color.rgb2lab(img))
			
			px_left = img_lab[0:border_thickness,:,:]
			# px_right = img_lab[n_rows - border_thickness-1:-1,:,:]
			px_right = img_lab[n_rows - border_thickness:, :, :]

			px_top = img_lab[:,0:border_thickness,:]
			# px_bottom = img_lab[:,n_cols - border_thickness-1:-1,:]
			px_bottom = img_lab[:, n_cols - border_thickness:, :]
			
			px_mean_left = np.mean(px_left,axis=(0,1))
			px_mean_right = np.mean(px_right,axis=(0,1))
			px_mean_top = np.mean(px_top,axis=(0,1))
			px_mean_bottom = np.mean(px_bottom,axis=(0,1))


			px_left = px_left.reshape((n_cols*border_thickness,3))
			px_right = px_right.reshape((n_cols*border_thickness,3))

			px_top = px_top.reshape((n_rows*border_thickness,3))
			px_bottom = px_bottom.reshape((n_rows*border_thickness,3))

			cov_left = np.cov(px_left.T)
			cov_right = np.cov(px_right.T)

			cov_top = np.cov(px_top.T)
			cov_bottom = np.cov(px_bottom.T)

			cov_left = np.linalg.inv(cov_left)
			cov_right = np.linalg.inv(cov_right)
			
			cov_top = np.linalg.inv(cov_top)
			cov_bottom = np.linalg.inv(cov_bottom)


			u_left = np.zeros(sal.shape)
			u_right = np.zeros(sal.shape)
			u_top = np.zeros(sal.shape)
			u_bottom = np.zeros(sal.shape)

			u_final = np.zeros(sal.shape)
			img_lab_unrolled = img_lab.reshape(img_lab.shape[0]*img_lab.shape[1],3)

			px_mean_left_2 = np.zeros((1,3))
			px_mean_left_2[0,:] = px_mean_left

			u_left = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_left_2,'mahalanobis', VI=cov_left)
			u_left = u_left.reshape((img_lab.shape[0],img_lab.shape[1]))

			px_mean_right_2 = np.zeros((1,3))
			px_mean_right_2[0,:] = px_mean_right

			u_right = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_right_2,'mahalanobis', VI=cov_right)
			u_right = u_right.reshape((img_lab.shape[0],img_lab.shape[1]))

			px_mean_top_2 = np.zeros((1,3))
			px_mean_top_2[0,:] = px_mean_top

			u_top = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_top_2,'mahalanobis', VI=cov_top)
			u_top = u_top.reshape((img_lab.shape[0],img_lab.shape[1]))

			px_mean_bottom_2 = np.zeros((1,3))
			px_mean_bottom_2[0,:] = px_mean_bottom

			u_bottom = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_bottom_2,'mahalanobis', VI=cov_bottom)
			u_bottom = u_bottom.reshape((img_lab.shape[0],img_lab.shape[1]))

			max_u_left = np.max(u_left)
			max_u_right = np.max(u_right)
			max_u_top = np.max(u_top)
			max_u_bottom = np.max(u_bottom)

			u_left = u_left / max_u_left
			u_right = u_right / max_u_right
			u_top = u_top / max_u_top
			u_bottom = u_bottom / max_u_bottom

			u_max = np.maximum(np.maximum(np.maximum(u_left,u_right),u_top),u_bottom)

			u_final = (u_left + u_right + u_top + u_bottom) - u_max

			u_max_final = np.max(u_final)
			sal_max = np.max(sal)
			sal = sal / sal_max + u_final / u_max_final

		#postprocessing

		# apply centredness map
		sal = sal / np.max(sal)
		
		s = np.mean(sal)
		alpha = 50.0
		delta = alpha * math.sqrt(s)

		xv,yv = np.meshgrid(np.arange(sal.shape[1]),np.arange(sal.shape[0]))
		(w,h) = sal.shape
		w2 = w/2.0
		h2 = h/2.0

		C = 1 - np.sqrt(np.power(xv - h2,2) + np.power(yv - w2,2)) / math.sqrt(np.power(w2,2) + np.power(h2,2))

		sal = sal * C

		#increase bg/fg contrast 

		def f(x):
			b = 10
			return 1.0 / (1.0 + math.exp(-b*(x - 0.5)))

		fv = np.vectorize(f)

		sal = sal / np.max(sal)

		sal = fv(sal)

		return sal* 255.0

def binarise_saliency_map(saliency_map,method='adaptive',threshold=0.5):

	# check if input is a numpy array
	if type(saliency_map).__module__ != np.__name__:
		print('Expected numpy array')
		return None

	#check if input is 2D
	if len(saliency_map.shape) != 2:
		print('Saliency map must be 2D')
		return None

	if method == 'fixed':
		return (saliency_map > threshold)

	elif method == 'adaptive':
		adaptive_threshold = 2.0 * saliency_map.mean()
		return (saliency_map > adaptive_threshold)

	elif method == 'clustering':
		print('Not yet implemented')
		return None

	else:
		print("Method must be one of fixed, adaptive or clustering")
		return None


# filename = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/sod_mbs/scripts/00.png'
filename = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/sod_mbs/scripts/dog.jpg'


# 检查文件是否存在
if not os.path.exists(filename):
    print(f"File does not exist: {filename}")
    exit()
# 原有代码
start_time = time.time()
mbd = get_saliency_mbd(filename).astype('uint8')
# 结束计时
end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
print(f"Saliency detection run time: {elapsed_time:.2f} seconds")

# 新增代码：生成二值显著图
binary_mbd = binarise_saliency_map(mbd, method='adaptive')
# 将布尔值转换为 uint8 类型，以便显示
binary_mbd = binary_mbd.astype('uint8') * 255

# 读取图片用于显示
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

cv2.imshow('img', img)
cv2.imshow('mbd', mbd)
cv2.imshow('binary_mbd', binary_mbd)  # 显示二值显著图
cv2.waitKey(0)
