import rospy
import struct
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

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
	# plt.imshow(intensity)
	# plt.show()
	# rospy.sleep(1)
	# plt.close()
	# cv2.imshow("Zeros matx", intensity) # show numpy array
	# cv2.waitKey(0) # wait for ay key to exit window
	# cv2.destroyAllWindows() # close all windows
	#print("Intensity BEV map generated.")
	return intensity

def publishImage(intensity, pub):
	bridge = CvBridge()
	image_message = bridge.cv2_to_imgmsg(intensity, encoding="rgb8")
	pub.publish(image_message)
	#print("Image published by ros node.")

# def drawBBox(image, bboxl):
# 	colour = (255, 0, 0)
# 	thickness = 2
# 	for i in range(len(bboxl)):
# 		startpt = (int(bboxl[i].minx), int(bboxl[i].miny))
# 		endpt = (int(bboxl[i].maxx), int(bboxl[i].maxy))
# 		image = cv2.rectangle(image, startpt, endpt, colour, thickness)
	
# 	cv2.imshow("window", image) 
	
class KittiSub(object):
	def __init__(self):
		#self.roi_sub = rospy.Subscriber('ROIL2D', BboxLes, self.roi_cb)
		self.pc_sub = rospy.Subscriber('/kitti_player/hdl64e', PointCloud2, self.cb, queue_size=1)
		self.pub = rospy.Publisher('/pixor_node/bev', Image, queue_size=1)
		self.scan = None#np.zeros((800, 700, 36), dtype=np.float32)
		self.density = np.zeros(800 * 700, dtype=np.int32)
		self.roiBox = None
		self.raw = None

	def roi_cb(self, roi):
		self.roiBox = roi
		
	def cb(self, data):
		t = time.time()
	
		self.raw = data
		self.scan = np.zeros((800, 700, 36), dtype=np.float32)
	
		def get_pts(x, y, z):
			r = ((int)((x - X_MIN)/X_DIVISION),
				 (int)((y - Y_MIN)/Y_DIVISION),
				 (int)((z - Z_MIN)/Z_DIVISION))
			return r
			
		def at2(a, b):
			return a*700 + b
			
		for p in pc2.read_points(self.raw, field_names = ("x", "y", "z", "intensity"), skip_nans=True):
		
			if(p[0] > X_MIN and p[1] > Y_MIN and p[2] > Z_MIN and p[0] < X_MAX and p[1] < Y_MAX and p[2] < Z_MAX):
				x, y, z = get_pts(p[0], p[1], p[2])
				
				self.scan[y, x, z] = 1
				self.scan[y, x, 35] += p[3]
				self.density[at2(y, x)] += 1	
		#normalization
		#for y in range(800):
		#	for x in range(700):
		#		if(self.density[at2(y,x)] > 0):
		#			self.scan[y, x, 35] = self.scan[y, x, 35] / self.density[at2(y, x)]
		intensity = get_intensity(self.scan)
		#drawBBox(intensity,self.roiBox.bboxl)
		publishImage(intensity,self.pub)
		dt = time.time() - t
		

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
	main()
