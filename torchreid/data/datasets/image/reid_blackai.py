from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings
import json
from ..dataset import ImageDataset
import random
from _collections import defaultdict
import torch
import torchvision
from torchvision import transforms, utils
from PIL import Image, ImageDraw
import torch.nn as nn
import numpy as np
import math
import cv2


def _xywh2cs( x, y, w, h):
    aspect_ratio = 192 * 1.0 / 256

    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / 200, h * 1.0 / 200],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def get_final_preds_using_softargmax(batch_heatmaps, conffilter, counterfilter):
    soft_argmax = SoftArgmax2D(64, 48, beta=160)
    coords, maxvals = soft_argmax(batch_heatmaps)

    output = []
    for idx, person in enumerate(maxvals):
         if len([i for i in person if i > conffilter]) > counterfilter:
             output.append(True)
         else:
             output.append(False)


    return output


    # heatmap_height = batch_heatmaps.shape[2]
    # heatmap_width = batch_heatmaps.shape[3]
	#
    # batch_heatmaps = batch_heatmaps.cpu().detach().numpy()
	#
	#
	#
    # # post-processing
    # for n in range(coords.shape[0]):
    #     for p in range(coords.shape[1]):
    #         hm = batch_heatmaps[n][p]
    #         px = int(math.floor(coords[n][p][0] + 0.5))
    #         py = int(math.floor(coords[n][p][1] + 0.5))
    #         if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
    #             diff = np.array(
    #         		[
    #         			hm[py][px + 1] - hm[py][px - 1],
    #         			hm[py + 1][px] - hm[py - 1][px]
    #         		]
    #         	)
    #             coords[n][p] += np.sign(diff) * .25
    # preds_ori = coords.copy()
    # preds = coords.copy()
	#
    # # Transform back
    # # for i in range(coords.shape[0]):
    # #     preds[i] = transform_preds(
    # #         coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
    # #     )
	#
    # return preds, maxvals, preds_ori

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def preparekeypointdataset(imagepaths):
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ]
	)

	imagedatalist = []
	for it in imagepaths:
		image = cv2.imread(it)
		c, s = _xywh2cs(0, 0, image.shape[1], image.shape[0])
		trans = get_affine_transform(c, s, 0, np.array([192, 256]))
		image = cv2.warpAffine(image,
		trans,
		(int(192), int(256)),
		flags=cv2.INTER_LINEAR)
		image = transform(image)
		#image_tra = torch.unsqueeze(transform(image), 0).cuda().half()
		imagedatalist.append(image.numpy())

	images_tensor = torch.from_numpy(np.stack(imagedatalist)).cuda().half()
	return images_tensor

def checkkeypoint( imagepath, model, filter = 0.6):
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),]
	)
	image = cv2.imread(imagepath)
	c, s = _xywh2cs(0, 0, image.shape[1], image.shape[0])
	trans = get_affine_transform(c, s, 0, np.array([192, 256]))
	image = cv2.warpAffine(
		image,
		trans,
		(int(192), int(256)),
		flags=cv2.INTER_LINEAR)

	image_tra = torch.unsqueeze(transform(image), 0).cuda().half()

	output = model(image_tra)

	return get_final_preds_using_softargmax(output, filter)



class SoftArgmax2D(nn.Module):
    def __init__(self, height=64, width=48, beta=100):
        super(SoftArgmax2D, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.beta = beta
        # Note that meshgrid in pytorch behaves differently with numpy.
        self.WY, self.WX = torch.meshgrid(torch.arange(height, dtype=torch.float),
                                          torch.arange(width, dtype=torch.float))


    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        probs = self.softmax(x.view(b, c, -1) * self.beta)
        probs = probs.view(b, c, h, w)

        self.WY = self.WY.to(device)
        self.WX = self.WX.to(device)

        px = torch.sum(probs * self.WX, dim=(2, 3))
        py = torch.sum(probs * self.WY, dim=(2, 3))
        preds = torch.stack((px, py), dim=-1).cpu().detach().numpy()

        idx = np.round(preds).astype(np.int32)
        maxvals = np.zeros(shape=(b, c, 1))
        for bi in range(b):
            for ci in range(c):
                maxvals[bi, ci, 0] = x[bi, ci, idx[bi, ci, 1], idx[bi, ci, 0]]

        return preds, maxvals

class reid_blackai_format(ImageDataset):
	"""blackai dataset

	the query images are selected at random for each of the uuid in the test set

	test
		--merge uuid
			-- detection uuid.png
	train
		--merge uuid
	"""
	_junk_pids = [0, -1]

	def __init__(self, root='', no_query_images = 4, **kwargs):
		self.no_query_images = no_query_images
		self.dataset_dir = root
		#self.model = torch.jit.load('/root/Blackai/lpn_pytorch_online/lpn_resnet50_trace.pt')
		self.model = None
		#self.download_dataset(self.dataset_dir, self.dataset_url)

		self.train_dir = osp.join(self.dataset_dir, 'train')
		self.test_dir = osp.join(self.dataset_dir, 'test')

		required_files = [
			self.train_dir,
			self.test_dir
		]
		self.check_before_run(required_files)
		print('training_set')
		self.train = self.process_dir(self.train_dir, False)
		print('gallery_set')
		self.gallery = self.process_dir(self.test_dir, False)
		self.query = self.select_query()

		json.dump(self.gallery, open("gallery_imgs.json", 'w'))
		json.dump(self.query, open("query_imgs.json", "w"))


		super(reid_blackai_format, self).__init__(self.train, self.query, self.gallery, **kwargs)

	def select_query(self):
		gallery = self.gallery
		dict = defaultdict(list)
		for (img_path, pid, camid) in gallery:
			dict[pid].append(img_path)

		query = []
		query_imgs = set()
		for pid, image_paths in dict.items():
			one_query = image_paths[:self.no_query_images]
			# one_query = random.choices(image_paths, k=self.no_query_images)
			camid = 1
			for img_path in one_query:
				query.append((img_path, pid, camid))
				query_imgs.add(img_path)
				camid += 1

		self.gallery = [(img_path, pid, camid) for (img_path, pid, camid) in gallery if not img_path in query_imgs]

		return query



	def process_dir(self, dir_path, keypointfilter):
		conffilter = 0.6
		countfilter = 13

		camid = 0
		uuids = os.listdir(dir_path)

		pid_container = set()
		for uuid in uuids:
			pid_container.add(uuid)
		pid2label = {pid: label for label, pid in enumerate(pid_container)}

		with open(dir_path.split("/")[-1]+"_ids.json", "w") as fp:
			json.dump(pid2label, fp)

		filter_data = []
		remain_rates = []

		for uuid in uuids:
			img_paths = os.listdir(osp.join(dir_path, uuid))
			file_list = []
			pid = pid2label[uuid]
			for img_path in img_paths:
				img_path = osp.join(dir_path, uuid, img_path)
				file_list.append(img_path)

			if keypointfilter:
				Images = preparekeypointdataset(file_list)
				output = self.model(Images)
				savepersons_idx = get_final_preds_using_softargmax(output, conffilter, countfilter)
				remain_rates.append(savepersons_idx.count(True) / len(savepersons_idx))

				for id, it in enumerate(file_list):
					if savepersons_idx[id]:
						filter_data.append((it, pid, camid))
			else:
				for id, it in enumerate(file_list):
					filter_data.append((it, pid, camid))
		print('conf_filter = {}, countfilter = {}'.format(conffilter, countfilter))
		print('remove_mean_rate = {}'.format(np.array(remain_rates).mean()))
		print('remove_std_rate = {}'.format(np.array(remain_rates).std()))


			# for img_path in img_paths:
			# 	img_path =osp.join(dir_path, uuid, img_path)
			# 	# if keypointfilter:
			# 	# 	correctnumber = checkkeypoint(img_path, self.model, filter = conffilter)
			# 	# 	if correctnumber < countfilter:
			# 	# 		continue
			# 	pid = pid2label[uuid]
			#
			# 	filter_data.append((img_path, pid, camid))

		return filter_data



