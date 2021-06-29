import rospy
import struct
import ctypes
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

from sensor_msgs.msg import PointCloud2, PointField
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

	return intensity
	
def publishImage(intensity, pub):
	bridge = CvBridge()
	image_message = bridge.cv2_to_imgmsg(intensity, encoding="rgb8")
	pub.publish(image_message)
	
class KittiSub(object):
	def __init__(self):
		self.pc_sub = rospy.Subscriber('/kitti_player/hdl64e', PointCloud2, self.cb, queue_size=1)
		self.pub = rospy.Publisher('/pixor_node/bev', Image, queue_size=1)
		self.scan = None#np.zeros((800, 700, 36), dtype=np.float32)
		self.density = np.zeros(800 * 700, dtype=np.int32)
		self.roiBox = None
		self.preprocess_lib = ctypes.cdll.LoadLibrary('./preprocess.so')

		self.raw = None
		
	def cb(self, data):
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
		dt = time.time() - t
		
		print("elapsed in {} seconds".format(dt))
		print(np.count_nonzero(self.scan))
		
		
		
def main():
	rospy.init_node('pixor_deal', anonymous=True)
	k = KittiSub()
	# rospy.sleep(4)
	# if (k.scan is not None):
		# intensity = get_intensity(k.scan)
		# for i in range(10):
		# 	publishImage(intensity,k.pub)
	rospy.spin()
	
if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass

