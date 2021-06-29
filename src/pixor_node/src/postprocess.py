import torch
import time
import numpy as np
from shapely.geometry import Polygon

def convert_format(boxes_array):
	"""

	:param array: an array of shape [# bboxs, 4, 2]
	:return: a shapely.geometry.Polygon object
	"""

	polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
	return np.array(polygons)
	
def compute_iou(box, boxes):
	"""Calculates IoU of the given box with the array of the given boxes.
	box: a polygon
	boxes: a vector of polygons
	Note: the areas are passed in rather than calculated here for
	efficiency. Calculate once in the caller to avoid duplicate work.
	"""
	# Calculate intersection areas
	iou = [box.intersection(b).area / box.union(b).area for b in boxes]

	return np.array(iou, dtype=np.float32)

def non_max_suppression(boxes, scores, threshold):
	"""Performs non-maximum suppression and returns indices of kept boxes.
	boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
	scores: 1-D array of box scores.
	threshold: Float. IoU threshold to use for filtering.

	return an numpy array of the positions of picks
	"""
	assert boxes.shape[0] > 0
	if boxes.dtype.kind != "f":
		boxes = boxes.astype(np.float32)

	polygons = convert_format(boxes)

	top = 64
	# Get indicies of boxes sorted by scores (highest first)
	ixs = scores.argsort()[::-1][:64]

	pick = []
	while len(ixs) > 0:
		# Pick top box and add its index to the list
		i = ixs[0]
		pick.append(i)
		# Compute IoU of the picked box with the rest
		iou = compute_iou(polygons[i], polygons[ixs[1:]])
		# Identify boxes with IoU over the threshold. This
		# returns indices into ixs[1:], so add 1 to get
		# indices into ixs.
		remove_ixs = np.where(iou > threshold)[0] + 1
		# Remove indices of the picked and overlapped boxes.
		ixs = np.delete(ixs, remove_ixs)
		ixs = np.delete(ixs, 0)

	return np.array(pick, dtype=np.int32)

#perform nonmax supression and filter predictions by difficulty
def filter_pred(config, pred):
	if len(pred.size()) == 4:
		if pred.size(0) == 1:
			pred.squeeze_(0)
		else:
			raise ValueError("Tensor dimension is not right")

	cls_pred = pred[0, ...]
	activation = cls_pred > config['cls_threshold']
	num_boxes = int(activation.sum())

	pred_boxes = {
		'easy': np.array([], dtype=np.float32),
		'medium': np.array([], dtype=np.float32),
		'hard': np.array([], dtype=np.float32)
	}

	pred_scores = {
		'easy': np.array([], dtype=np.float32),
		'medium': np.array([], dtype=np.float32),
		'hard': np.array([], dtype=np.float32)
	}

	if num_boxes == 0:
		#print("No bounding box found")
		return pred_boxes, pred_scores

	corners = torch.zeros((num_boxes, 8))
	for i in range(7, 15):
		corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
		
	corners = corners.view(-1, 4, 2).numpy()
	scores = torch.masked_select(cls_pred, activation).cpu().numpy()

	# NMS
	#t_0 = time.time()
	selected_ids = non_max_suppression(corners, scores, config['nms_iou_threshold'])
	#t_nms = time.time() - t_0

	#corners = corners[selected_ids]
	#scores = scores[selected_ids]

	for id in selected_ids:
		#get the midpoint of the prediction box, note: if some predictions
		#are on the edge of easy/medium or medium/hard, they may get misclassified
		#due to slight offsets in the prediction. 
		cx = corners[id][:, 0].sum()/4
		cy = corners[id][:, 1].sum()/4
		dist = np.sqrt(cx*cx + cy*cy) 

		if dist >= 50:
			pred_boxes['hard'] = np.append(pred_boxes['hard'], corners[id])
			pred_scores['hard'] = np.append(pred_scores['hard'], scores[id])
		elif dist < 50 and dist >= 30:
			pred_boxes['medium'] = np.append(pred_boxes['medium'], corners[id])
			pred_scores['medium'] = np.append(pred_scores['medium'], scores[id])
		else:
			pred_boxes['easy'] = np.append(pred_boxes['easy'], corners[id])
			pred_scores['easy'] = np.append(pred_scores['easy'], scores[id])

	#reshape the prediction boxes since np.append changes the shape
	for k in pred_boxes:
		pred_boxes[k] = pred_boxes[k].reshape(-1, 4, 2)

	return pred_boxes, pred_scores
