import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
	def __init__(self, device, config, num_classes=1):
		super(CustomLoss, self).__init__()
		self.num_classes = num_classes
		self.device = device
		self.alpha = config['alpha']
		self.beta = config['beta']
		self.gamma = config['gamma']

	def cross_entropy(self, x, y):
		return F.binary_cross_entropy(input=x, target=y, reduction='elementwise_mean')


	def forward(self, preds, targets):
		'''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
		Args:
		preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
		cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
		cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
		loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
		loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
		loss:
		(tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
		'''

		#target_classes = targets[:, 7, :, :] #like this because of (2, 0, 1) permutation
		targets = targets[:, :7 :, :]

		batch_size = targets.size(0)
		image_size = targets.size(1) * targets.size(2)
		cls_targets, loc_targets = targets.split([1, 6], dim=1)
		if preds.size(1) == 7:
			cls_preds, loc_preds = preds.split([1, 6], dim=1)
		elif preds.size(1) == 15:
			cls_preds, loc_preds, _ = preds.split([1, 6, 8], dim=1)

		#Compute BCE loss
		cls_loss = self.cross_entropy(cls_preds, cls_targets) * self.alpha

		#Determine pt
		pt = torch.exp(-cls_loss)

		#Compute focal loss
		focal_loss = (((1 - pt)**self.gamma) * cls_loss).mean()
		cls = focal_loss.item()

		pos_pixels = cls_targets.sum()
		if pos_pixels > 0:
			loc_loss = F.smooth_l1_loss(cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
			loc = loc_loss.item()
			loss = loc_loss + cls_loss
		else:
			loc = 0.0
			loss = cls_loss

		#print(cls, loc)
		return loss, cls, loc
