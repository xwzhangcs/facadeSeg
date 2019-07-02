#######################################################################
# Generate HTML file that shows the input and output images to compare.

from os.path import isfile, join
from PIL import Image
import os
import argparse
import numpy as np
import json
import subprocess
import sys
import shutil
import glob
import pandas as pd
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage import io
import cv2
import random

train = 240
val = 80
test = 50

def collect_segs(aoi_dir, data):
	aoi_areas = sorted(os.listdir(aoi_dir))
	real_facade_dir = data + '/A'
	real_facade_dir_train = data + '/A/train'
	real_facade_dir_val = data + '/A/val'
	real_facade_dir_test = data + '/A/test'
	seg_facade_dir = data + '/B'
	seg_facade_dir_train = data + '/B/train'
	seg_facade_dir_val = data + '/B/val'
	seg_facade_dir_test = data + '/B/test'

	if not os.path.exists(real_facade_dir):
		os.makedirs(real_facade_dir)
		os.makedirs(real_facade_dir_train)
		os.makedirs(real_facade_dir_val)
		os.makedirs(real_facade_dir_test)

	if not os.path.exists(seg_facade_dir):
		os.makedirs(seg_facade_dir)
		os.makedirs(seg_facade_dir_train)
		os.makedirs(seg_facade_dir_val)
		os.makedirs(seg_facade_dir_test)

	index = 0
	for l in range(len(aoi_areas)):
		clusters = sorted(os.listdir(aoi_dir + '/' + aoi_areas[l]))
		for i in range(len(clusters)):
			cluster = aoi_dir + '/' + aoi_areas[l] + '/' + clusters[i]
			if not os.path.exists(cluster + '/label'):
				continue
			seg_images = sorted(os.listdir(cluster + '/seg'))
			for j in range(len(seg_images)):
				facade_img_a_name = cluster + '/label/' + seg_images[j][: len(seg_images[j]) - 4] + '_A.png'
				facade_img_b_name = cluster + '/label/' + seg_images[j][: len(seg_images[j]) - 4] + '_B.png'
				if 0 <= index < train:
					# resize
					real_facade_name = real_facade_dir_train + '/facade_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 0)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 0)
					# horizontal flip
					real_facade_name = real_facade_dir_train + '/facade_h_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_h_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 1)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 1)
					# vertical flip
					real_facade_name = real_facade_dir_train + '/facade_v_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_v_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 2)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 2)
					# both flip / rotate 180
					real_facade_name = real_facade_dir_train + '/facade_t_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_t_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 3)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 3)
					# rotate 90
					real_facade_name = real_facade_dir_train + '/facade_lr_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_lr_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 4)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 4)
					# rotate 270
					real_facade_name = real_facade_dir_train + '/facade_rr_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_rr_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 5)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 5)
					# increase brightness
					real_facade_name = real_facade_dir_train + '/facade_inc_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_inc_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 6)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 6)
					# decrease brightness
					real_facade_name = real_facade_dir_train + '/facade_dec_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_train + '/facade_dec_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 7)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 7)
				elif train <= index < train + val:
					real_facade_name = real_facade_dir_val + '/facade_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 0)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 0)
					# horizontal flip
					real_facade_name = real_facade_dir_val + '/facade_h_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_h_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 1)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 1)
					# vertical flip
					real_facade_name = real_facade_dir_val + '/facade_v_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_v_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 2)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 2)
					# both flip / rotate 180
					real_facade_name = real_facade_dir_val + '/facade_t_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_t_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 3)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 3)
					# rotate 90
					real_facade_name = real_facade_dir_val + '/facade_lr_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_lr_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 4)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 4)
					# rotate 270
					real_facade_name = real_facade_dir_val + '/facade_rr_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_rr_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 5)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 5)
					# increase brightness
					real_facade_name = real_facade_dir_val + '/facade_inc_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_inc_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 6)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 6)
					# decrease brightness
					real_facade_name = real_facade_dir_val + '/facade_dec_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_val + '/facade_dec_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 7)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 7)
				else:
					real_facade_name = real_facade_dir_test + '/facade_' + format(index, '05d') + '.png'
					seg_facade_name = seg_facade_dir_test + '/facade_' + format(index, '05d') + '.png'
					data_augmentation(facade_img_a_name, real_facade_name, 286, 0)
					data_augmentation(facade_img_b_name, seg_facade_name, 286, 0)
				index = index + 1


def crop_segs(aoi_dir):
	aoi_areas = sorted(os.listdir(aoi_dir))
	for l in range(len(aoi_areas)):
		clusters = sorted(os.listdir(aoi_dir + '/' + aoi_areas[l]))
		for i in range(len(clusters)):
			cluster = aoi_dir + '/' + aoi_areas[l] + '/' + clusters[i]
			if not os.path.exists(cluster + '/seg'):
				continue
			if not os.path.exists(cluster + '/label'):
				os.makedirs(cluster + '/label')
			seg_images = sorted(os.listdir(cluster + '/seg'))
			for j in range(len(seg_images)):
				facade_img_name = cluster + '/image/' + seg_images[j]
				seg_img_name = cluster + '/seg/' + seg_images[j]
				facade_img_a_name = cluster + '/label/' + seg_images[j][: len(seg_images[j]) - 4] + '_A.png'
				facade_img_b_name = cluster + '/label/' + seg_images[j][: len(seg_images[j]) - 4] + '_B.png'
				# find the rectangle
				seg_img = cv2.imread(seg_img_name, cv2.IMREAD_UNCHANGED)
				facade_img = cv2.imread(facade_img_name, cv2.IMREAD_UNCHANGED)
				height, width = seg_img.shape[:2]
				# correct seg_img
				for x in range(height):
					for y in range(width):
						#if not (seg_img[x][y][0] == 255 and seg_img[x][y][1] == 0 and seg_img[x][y][2] == 0) and not (seg_img[x][y][0] == 0 and seg_img[x][y][1] == 0 and seg_img[x][y][2] == 255):
						if seg_img[x][y][1] != 0:
							seg_img[x][y][0] = 255
							seg_img[x][y][1] = 0
							seg_img[x][y][2] = 0
				top = 0
				bot = 0
				left = 0
				right = 0
				padding = 5

				bTop = False
				for x in range(height):
					if bTop:
						break
					for y in range(width):
						if seg_img[x][y][2] == 255:
							top = x
							bTop = True
							break
				if top - padding < 0:
					top = 0
				else:
					top = top - padding

				bBot = False
				for x in range(height - 1, 0, -1):
					if bBot:
						break
					for y in range(width):
						if seg_img[x][y][2] == 255:
							bBot = True
							bot = x
							break
				if bot + padding > height - 1:
					bot = height - 1
				else:
					bot = bot + padding

				bLeft = False
				for y in range(width):
					if bLeft:
						break
					for x in range(height):
						if seg_img[x][y][2] == 255:
							left = y
							bLeft = True
							break
				if left - padding < 0:
					left = 0
				else:
					left = left - padding

				bRight = False
				for y in range(width - 1, 0, -1):
					if bRight:
						break
					for x in range(height):
						if seg_img[x][y][2] == 255:
							right = y
							bRight = True
							break
				if right + padding > width - 1:
					right = width - 1
				else:
					right = right + padding

				print(left, right, top, bot)
				facade_crop_img = facade_img[top:bot, left:right]
				seg_crop_img = seg_img[top:bot, left:right]
				cv2.imwrite(facade_img_a_name, facade_crop_img)
				cv2.imwrite(facade_img_b_name, seg_crop_img)


def data_augmentation(src_imgagename, out_imagename, target_size, type):
	src_img = cv2.imread(src_imgagename, cv2.IMREAD_UNCHANGED)
	resize_img = cv2.resize(src_img, (target_size, target_size))
	horizontal_img = resize_img.copy()
	vertical_img = resize_img.copy()
	both_img = resize_img.copy()
	rotate_90_img = resize_img.copy()
	rotate_270_img = resize_img.copy()
	bright_img = resize_img.copy()
	dark_img = resize_img.copy()

	if type == -1:
		cv2.imwrite(out_imagename, src_img)
	elif type == 0:
		cv2.imwrite(out_imagename, resize_img)
	elif type == 1: # horizontal flip
		horizontal_img = cv2.flip(resize_img, 1)
		cv2.imwrite(out_imagename, horizontal_img)
	elif type == 2: # vertical flip
		vertical_img = cv2.flip(resize_img, 0)
		cv2.imwrite(out_imagename, vertical_img)
	elif type == 3: # both flip / rotate 180
		both_img = cv2.flip(resize_img, -1)
		cv2.imwrite(out_imagename, both_img)
	elif type == 4: # rotate 90
		row, col = resize_img.shape[:2]
		center = tuple(np.array([row, col]) / 2)
		angle = 90
		rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotate_90_img = cv2.warpAffine(resize_img, rot_mat, (col, row))
		cv2.imwrite(out_imagename, rotate_90_img)
	elif type == 5: # rotate 270
		row, col = resize_img.shape[:2]
		center = tuple(np.array([row, col]) / 2)
		angle = -90
		rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotate_270_img = cv2.warpAffine(resize_img, rot_mat, (col, row))
		cv2.imwrite(out_imagename, rotate_270_img)
	elif type == 6: # increase brightness
		increase = 30
		image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
		v = image[:, :, 2]
		v = np.where(v <= 255 - increase, v + increase, 255)
		image[:, :, 2] = v
		bright_img = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
		cv2.imwrite(out_imagename, bright_img)
	elif type == 7: # decrease brightness
		decrease = 30
		image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
		v = image[:, :, 2]
		v = np.where(v > decrease, v - decrease, 0)
		image[:, :, 2] = v
		dark_img = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
		cv2.imwrite(out_imagename, dark_img)


def data_shuffle(src_path, output_path, type):
	real_facade_dir = output_path + '/A'
	real_facade_dir_train = output_path + '/A/train'
	real_facade_dir_val = output_path + '/A/val'
	real_facade_dir_test = output_path + '/A/test'
	seg_facade_dir = output_path + '/B'
	seg_facade_dir_train = output_path + '/B/train'
	seg_facade_dir_val = output_path + '/B/val'
	seg_facade_dir_test = output_path + '/B/test'

	if not os.path.exists(real_facade_dir):
		os.makedirs(real_facade_dir)
		os.makedirs(real_facade_dir_train)
		os.makedirs(real_facade_dir_val)
		os.makedirs(real_facade_dir_test)

	if not os.path.exists(seg_facade_dir):
		os.makedirs(seg_facade_dir)
		os.makedirs(seg_facade_dir_train)
		os.makedirs(seg_facade_dir_val)
		os.makedirs(seg_facade_dir_test)

	# shuffle train
	if type == 0:
		src_train = src_path + '/A/train'
		src_train_images = sorted(os.listdir(src_train))
		src_train_dic = dict(zip(list(range(len(src_train_images))), src_train_images))
		l = list(src_train_dic.items())
		random.shuffle(l)
		src_train_dic = dict(l)
		index = 0
		for key, value in src_train_dic.items():
			src_train_facade_img = src_train + '/' + value
			src_train_seg_img = src_path + '/B/train/' + value
			output_train_facade_img = real_facade_dir_train + '/facade_' + format(index, '05d') + '.png'
			output_train_seg_img = seg_facade_dir_train + '/facade_' + format(index, '05d') + '.png'
			src_img = cv2.imread(src_train_facade_img, cv2.IMREAD_UNCHANGED)
			cv2.imwrite(output_train_facade_img, src_img)
			src_img = cv2.imread(src_train_seg_img, cv2.IMREAD_UNCHANGED)
			cv2.imwrite(output_train_seg_img, src_img)
			index = index + 1
	# shuffle val
	if type == 1:
		src_val= src_path + '/A/val'
		src_val_images = sorted(os.listdir(src_val))
		src_val_dic = dict(zip(list(range(len(src_val_images))), src_val_images))
		l = list(src_val_dic.items())
		random.shuffle(l)
		src_val_dic = dict(l)
		index = 0
		for key, value in src_val_dic.items():
			src_val_facade_img = src_val + '/' + value
			src_val_seg_img = src_path + '/B/val/' + value
			output_val_facade_img = real_facade_dir_val + '/facade_' + format(index, '05d') + '.png'
			output_val_seg_img = seg_facade_dir_val + '/facade_' + format(index, '05d') + '.png'
			src_img = cv2.imread(src_val_facade_img, cv2.IMREAD_UNCHANGED)
			cv2.imwrite(output_val_facade_img, src_img)
			src_img = cv2.imread(src_val_seg_img, cv2.IMREAD_UNCHANGED)
			cv2.imwrite(output_val_seg_img, src_img)
			index = index + 1
	if type == 2:
		src_test = src_path + '/A/test'
		src_test_images = sorted(os.listdir(src_test))
		src_test_dic = dict(zip(list(range(len(src_test_images))), src_test_images))
		index = 0
		for key, value in src_test_dic.items():
			src_test_facade_img = src_test + '/' + value
			src_test_seg_img = src_path + '/B/test/' + value
			output_test_facade_img = real_facade_dir_test + '/facade_' + format(index, '05d') + '.png'
			output_test_seg_img = seg_facade_dir_test + '/facade_' + format(index, '05d') + '.png'
			src_img = cv2.imread(src_test_facade_img, cv2.IMREAD_UNCHANGED)
			cv2.imwrite(output_test_facade_img, src_img)
			src_img = cv2.imread(src_test_seg_img, cv2.IMREAD_UNCHANGED)
			cv2.imwrite(output_test_seg_img, src_img)
			index = index + 1


def main(aoi_dir, data):
	# crop_segs(aoi_dir)
	# collect_segs(aoi_dir, data)
	data_shuffle('data/data_augmentation', 'data/output', 0)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("aoi_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("data", help="path to input image folder (e.g., input_data)")
	args = parser.parse_args()

	main(args.aoi_dir, args.data)
