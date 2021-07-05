import rospy
import struct
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import pcl_helper as helper

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
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

def get_intensity(scan):
	intensity = np.zeros((scan.shape[0], scan.shape[1], 3), dtype=np.uint8)   
	val = (1 - np.amax(scan, axis=2)) * 255
	
	intensity[:, :, 0] = val
	intensity[:, :, 1] = val
	intensity[:, :, 2] = val
	print("Intensity BEV map generated.")
	return intensity

def publishImage(intensity, pub):
	bridge = CvBridge()
	image_message = bridge.cv2_to_imgmsg(intensity, encoding="rgb8")
	pub.publish(image_message)
	print("Image published by ros node.\n")

def drawBBox(image, roiPts):
	colour = (255, 0, 0)
	thickness = 2
	for p in pc2.read_points(roiPts, field_names=("x", "y", "z", "rgb"), skip_nans=True):
		rgb = helper.float_to_rgb(p[3])
		xmin, ymin, zmin = get_pts(p[1], rgb[0]*(rgb[2]-1), -1.7)
		xmax, ymax, zmax = get_pts(p[0], p[2], rgb[1]/10-10)
		del_x = xmax-xmin
		del_y = ymax-ymin
		if del_x and del_y !=0:
			ratio = del_x/del_y
			if 4.5 <= ratio <= 5.5:
				scale = del_x/12
				#if 9 >= scale >= 7.5:
				#	image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), colour, thickness)
			elif 1 <= ratio <= 5:
				scale = del_x/4
				if 4 <= scale <= 12:
					image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), colour, thickness)
	
	print("Bounding Boxes Drawn.")
		
def get_pts(x, y, z):
			r = ((int)((x - X_MIN)/X_DIVISION),
				 (int)((y - Y_MIN)/Y_DIVISION),
				 (int)((z - Z_MIN)/Z_DIVISION))
			return r
	
class KittiSub(object):
	def __init__(self):
		self.roi_sub = rospy.Subscriber('ROI', PointCloud2, self.roi_cb)
		self.pc_sub = rospy.Subscriber('/kitti_player/hdl64e', PointCloud2, self.cb, queue_size=1)
		self.pub = rospy.Publisher('/pixor_node/bev', Image, queue_size=1)
		self.scan = None#np.zeros((800, 700, 36), dtype=np.float32)
		self.density = np.zeros(800 * 700, dtype=np.int32)
		self.roiPts = None
		self.raw = None

	def roi_cb(self, roi):
		self.roiPts = roi
		
	def cb(self, data):
	
		self.raw = data
		self.scan = np.zeros((800, 700, 36), dtype=np.float32)
			
		def at2(a, b):
			return a*700 + b
			
		for p in pc2.read_points(self.raw, field_names = ("x", "y", "z", "intensity"), skip_nans=True):
		
			if(p[0] > X_MIN and p[1] > Y_MIN and p[2] > Z_MIN and p[0] < X_MAX and p[1] < Y_MAX and p[2] < Z_MAX):
				x, y, z = get_pts(p[0], p[1], p[2])
				
				self.scan[y, x, z] = 1
				self.scan[y, x, 35] += p[3]
				self.density[at2(y, x)] += 1	

		intensity = get_intensity(self.scan)
		if self.roiPts is not None:
			drawBBox(intensity,self.roiPts)
		
		publishImage(intensity,self.pub)
		

def main():
	rospy.init_node('pixor_deal', anonymous=True)
	k = KittiSub()

	rospy.spin()
if __name__ == '__main__':
	main()
