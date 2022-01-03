import ctypes
import math
import cv2
import os
import json
import time
import argparse
import numpy as np
np.set_printoptions(suppress=True)

#PyTorch imports
import torch
import torch.nn as nn

#PIXOR imports
from model import PIXOR
from loss import CustomLoss
from postprocess import filter_pred

#ROS imports
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from opencv_deal.msg import BboxLes, BboxL

X_MIN = 0.0;
X_MAX = 70.0;
Y_MIN =-40.0;
Y_MAX = 40.0;
Z_MIN = -2.5; 
Z_MAX = 1;
X_DIVISION = 0.1;
Y_DIVISION = 0.1;
Z_DIVISION = 0.1;

def project_to_image(points, proj_mat):
	"""
	Apply the perspective projection
	Args:
		pts_3d:     3D points in camera coordinate [3, npoints]
		proj_mat:   Projection matrix [3, 4]
	"""
	num_pts = points.shape[1]

	# Change to homogenous coordinate
	points = np.vstack((points, np.ones((1, num_pts))))
	points = proj_mat @ points
	points[:2, :] /= points[2, :]
	return points[:2, :]

def get_points_in_a_rotated_box(corners, label_shape=[200, 175]):
	def minY(x0, y0, x1, y1, x):
		if x0 == x1:
			# vertical line, y0 is lowest
			return int(math.floor(y0))

		m = (y1 - y0) / (x1 - x0)

		if m >= 0.0:
			# lowest point is at left edge of pixel column
			return int(math.floor(y0 + m * (x - x0)))
		else:
			# lowest point is at right edge of pixel column
			return int(math.floor(y0 + m * ((x + 1.0) - x0)))


	def maxY(x0, y0, x1, y1, x):
		if x0 == x1:
			# vertical line, y1 is highest
			return int(math.ceil(y1))

		m = (y1 - y0) / (x1 - x0)

		if m >= 0.0:
			# highest point is at right edge of pixel column
			return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
		else:
			# highest point is at left edge of pixel column
			return int(math.ceil(y0 + m * (x - x0)))


	# view_bl, view_tl, view_tr, view_br are the corners of the rectangle
	view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

	pixels = []

	# find l,r,t,b,m1,m2
	l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
	b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

	lx, ly = l
	rx, ry = r
	bx, by = b
	tx, ty = t
	m1x, m1y = m1
	m2x, m2y = m2

	xmin = 0
	ymin = 0
	xmax = label_shape[1]
	ymax = label_shape[0]

	# inward-rounded integer bounds
	# note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
	lxi = max(int(math.ceil(lx)), xmin)
	rxi = min(int(math.floor(rx)), xmax)
	byi = max(int(math.ceil(by)), ymin)
	tyi = min(int(math.floor(ty)), ymax)

	x1 = lxi
	x2 = rxi

	for x in range(x1, x2):
		xf = float(x)

		if xf < m1x:
			# Phase I: left to top and bottom
			y1 = minY(lx, ly, bx, by, xf)
			y2 = maxY(lx, ly, tx, ty, xf)

		elif xf < m2x:
			if m1y < m2y:
				# Phase IIa: left/bottom --> top/right
				y1 = minY(bx, by, rx, ry, xf)
				y2 = maxY(lx, ly, tx, ty, xf)

			else:
				# Phase IIb: left/top --> bottom/right
				y1 = minY(lx, ly, bx, by, xf)
				y2 = maxY(tx, ty, rx, ry, xf)

		else:
			# Phase III: bottom/top --> right
			y1 = minY(bx, by, rx, ry, xf)
			y2 = maxY(tx, ty, rx, ry, xf)

		y1 = max(y1, byi)
		y2 = min(y2, tyi)

		for y in range(y1, y2):
			pixels.append((x, y))

	return pixels

def load_config(exp_name):
	""" Loads the configuration file

	Args:
	path: A string indicating the path to the configuration file
	Returns:
	config: A Python dictionary of hyperparameter name-value pairs
	learning rate: The learning rate of the optimzer
	batch_size: Batch size used during training
	num_epochs: Number of epochs to train the network for
	target_classes: A list of strings denoting the classes to
		        build the classifer for
	"""
	path = os.path.join('experiments', exp_name, 'config.json')
	with open(path) as file:
		config = json.load(file)

	assert config['name']==exp_name

	learning_rate = config["learning_rate"]
	batch_size = config["batch_size"]
	max_epochs = config["max_epochs"]

	return config, learning_rate, batch_size, max_epochs
	
def get_model_name(config, epoch=None):
	""" Generate a name for the model consisting of all the hyperparameter values

	Args:
	name: Name of ckpt
	Returns:
	path: A string with the hyperparameter name and value concatenated
	"""
	# path = "model_"
	# path += "epoch{}_".format(config["max_epochs"])
	# path += "bs{}_".format(config["batch_size"])
	# path += "lr{}".format(config["learning_rate"])

	name = config['name']
	if epoch is None:
		epoch = config['resume_from']

	folder = os.path.join("experiments", name)
	if not os.path.exists(folder):
		os.makedirs(folder)

	path = os.path.join(folder, str(epoch)+"epoch")
	print(path)
	return path

def build_model(config, device, train=True):
	net = PIXOR(config['geometry'], config['use_bn'], device)
	loss_fn = CustomLoss(device, config, num_classes=1)

	if torch.cuda.device_count() <= 1:
		config['mGPUs'] = False
	if config['mGPUs']:
		print("using multi gpu")
		net = nn.DataParallel(net)

	net = net.to(device)
	loss_fn = loss_fn.to(device)
	if not train:
		return net, loss_fn

	optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_decay_at'], gamma=0.1)

	return net, loss_fn, optimizer, scheduler

def get_bev(scan, label_list, BboxArr, proj_mat):
	map_height = scan.shape[0]
	intensity = np.zeros((scan.shape[0], scan.shape[1], 3), dtype=np.uint8)
	
	z_min = -1.5
	z_max = -0
	val = (1 - scan[::-1, :, :-1].max(axis=2)) * 255
	
	intensity[:, :, 0] = val
	intensity[:, :, 1] = val
	intensity[:, :, 2] = val
	
	if label_list is not None:
		for difficulty in label_list:
			#set colour
			if difficulty == 'easy':
				colour = (255,0, 0)
			elif difficulty == 'medium':
				colour = (0, 165, 255)
			else:
				colour = (0, 0, 255)
				
			for corners in label_list[difficulty]:	
				#get location in output map
				label_corners = (corners/ 4 ) / 0.1
				label_corners[:, 1] += 200 / 2 #geometry of label_shape[0] 
				points = get_points_in_a_rotated_box(label_corners)

				#plot the bounding box
				plot_corners = corners / 0.1
				plot_corners[:, 1] += int(map_height // 2)
				plot_corners[:, 1] = map_height - plot_corners[:, 1]

				plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
				
				if difficulty == "easy" or "medium":
					minx, miny, maxx, maxy = min(plot_corners[:,:,0])[0]/10, min(-(plot_corners[:,:,1]-400))[0]/10, max(plot_corners[:,:,0])[0]/10, max(-(plot_corners[:,:,1]-400))[0]/10
			
					new_Bbox = BboxL()
					new_Bbox.maxx, new_Bbox.maxy = int(maxx), int(maxy)
					new_Bbox.minx, new_Bbox.miny = int(minx), int(miny)
					BboxArr.bboxl.append(new_Bbox)
					print(minx, miny, maxx, maxy)
					cam = np.array([[minx,minx,maxx,maxx, minx,minx,maxx,maxx],
									[miny,maxy,miny,maxy, miny,maxy,miny,maxy],
									[z_min,z_min,z_min,z_min, z_max,z_max,z_max,z_max]])
					cam = project_to_image(cam,proj_mat)
					print(cam)
					xval = np.delete(cam[0,:],np.where(cam[0,:]<0))
					yval = np.delete(cam[1,:],np.where(cam[1,:]<0))
					if len(xval) and len(yval) > 1:
						minx, miny, maxx, maxy = min(xval), min(yval), max(xval), max(yval)

						new_Bbox = BboxL()
						new_Bbox.maxx, new_Bbox.maxy = maxx, maxy
						new_Bbox.minx, new_Bbox.miny = minx, miny

						BboxArr.bboxl.append(new_Bbox)

				cv2.polylines(intensity, [plot_corners], True, colour, 2)
				cv2.line(intensity, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 3)

				tx = int((plot_corners[3][0][0] + plot_corners[1][0][0]) / 2) - 10
				ty = int((plot_corners[3][0][1] + plot_corners[1][0][1]) / 2) - 15
				cv2.putText(intensity, 'car', (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	return intensity
	
def publishImage(intensity, pub):
	bridge = CvBridge()
	image_message = bridge.cv2_to_imgmsg(intensity, encoding="rgb8")
	pub.publish(image_message)
	
class KittiSub(object):
	def __init__(self, name, device):
		self.ready = False
		self.pc_sub = rospy.Subscriber('/kitti_player/hdl64e', PointCloud2, self.cb, queue_size=1)
		self.pub = rospy.Publisher('/pixor_node/bev', Image, queue_size=1)
		self.bbox_pub = rospy.Publisher("/pixor_node/bbox",BboxLes, queue_size=1)
		self.scan = None#np.zeros((800, 700, 36), dtype=np.float32)
		self.density = np.zeros(800 * 700, dtype=np.int32)
		self.roiBox = None
		self.preprocess_lib = ctypes.cdll.LoadLibrary('./preprocessing/preprocess.so')

		self.raw = None
		
		#initialize PIXOR network
		self.device = device
		
		self.config, _, _, _ = load_config(name)
		self.net, self.loss_fn = build_model(self.config, self.device, train=False)
		state_dict = torch.load(get_model_name(self.config), map_location=device)
		self.net.eval()
		
		#load network weights
		if self.config['mGPUs']:
			self.net.module.set_decode(True)
			self.net.module.load_state_dict(state_dict)
		else:
			self.net.set_decode(True)
			self.net.load_state_dict(state_dict)
			
		self.ready = True
		
	def cb(self, data):
		if self.ready:
			t = time.time()
		
			self.raw = data
			self.scan = np.zeros((800, 700, 36), dtype=np.float32)	
			
			big = self.raw.is_bigendian
			fields = self.raw.fields
			field_names = ("x", "y", "z", "intensity")
			fmt = '<fffxxxxf'
		    	
			c_nparray = ctypes.c_void_p(self.scan.ctypes.data)
			c_pcdata = ctypes.create_string_buffer(self.raw.data, len(self.raw.data))
			c_height = ctypes.c_uint32(self.raw.height)
			c_width = ctypes.c_uint32(self.raw.width)
			c_point_step = ctypes.c_uint32(self.raw.point_step)
			c_row_step = ctypes.c_uint32(self.raw.row_step)
			self.preprocess_lib.processBEV(c_nparray, c_pcdata, c_height, c_width, c_point_step, c_row_step)
			
			dt1 = time.time()
			
			t_preprocess = time.time()
			dt_preprocess = t_preprocess - t
			# print("preprocess elapsed in {} seconds".format(dt_preprocess))
			
			#format scan
			input_np = self.scan
			self.scan = torch.from_numpy(self.scan)
			self.scan = self.scan.permute(2, 0, 1)
			
			#get PIXOR predictions
			preds = {
				'easy': np.array([], dtype=np.float32),
				'medium': np.array([], dtype=np.float32),
				'hard': np.array([], dtype=np.float32)
			}

			with torch.no_grad():
				input = torch.reshape(self.scan, (1, 36, 800, 700))
				input = input.to(self.device)
				
				dt2 = time.time()
				# print("Network input formatting completed in {} seconds".format(dt2 - dt1))
				
				# t_in_format = time.time()
				# dt_in_format = t_in_format - t_preprocess
				# print("Input Formatting elapsed in {} seconds".format(dt_in_format)) 
				
				predictions = self.net(input)
				
				dt3 = time.time()
				# print("Network prediction completed in {} seconds".format(dt3 - dt2))
				
				# This is a major bottleneck (~100ms)
				for pred in predictions:
					corners, scores = filter_pred(self.config, pred)

					for k in corners:
						for corner in corners[k]:
							preds[k] = np.append(preds[k], corner)
							
				dt4 = time.time()
				# print("prediction filtering completed in {} seconds".format(dt4 - dt3))
				
				#this takes no time
				for k in preds:
					preds[k] = preds[k].reshape(-1, 4, 2)
					
					
			# dt_network = time.time() - t_in_format
			# print("Network elapsed in {} seconds".format(dt_network)) 
			
			dt = time.time()
			# print("Completed in {} seconds \n".format(dt - t))
			BboxArr = BboxLes()
			proj_mat = np.array([
				[609.6954, -721.4216,   -1.2513, -123.0418],
				[180.3842,  7.6448, -719.6515, -101.0167],
    			[0.9999,    0.0001,    0.0105,   -0.2694]
			])
			intensity = get_bev(input_np, preds, BboxArr, proj_mat)
			publishImage(intensity, self.pub)
			self.bbox_pub.publish(BboxArr)
		
def main(name, device):
	rospy.init_node('pixor_deal', anonymous=True)
	k = KittiSub(name, device)
	rospy.spin()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PIXOR runtime test')
	parser.add_argument('--name', required=True, help="name of the experiment")
	parser.add_argument('--device', default='cpu', help='device to train on')
	args = parser.parse_args()

	name = args.name
	device = torch.device(args.device)
	
	if not torch.cuda.is_available():
		device = torch.device('cpu')
	print("Using device", device)
	
	try:
		main(name, device)
	except rospy.ROSInterruptException:
		pass
