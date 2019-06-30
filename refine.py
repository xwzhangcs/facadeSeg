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

def collect_segs(aoi_dir, data):
	aoi_areas = sorted(os.listdir(aoi_dir))
	real_facade_dir = data + '/A'
	seg_facade_dir = data + '/B'
	if not os.path.exists(real_facade_dir):
		os.makedirs(real_facade_dir)
	if not os.path.exists(seg_facade_dir):
		os.makedirs(seg_facade_dir)

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
				seg_img = cv2.imread(facade_img_a_name, cv2.IMREAD_UNCHANGED)
				facade_img = cv2.imread(facade_img_b_name, cv2.IMREAD_UNCHANGED)
				real_facade_name = real_facade_dir + '/facade_' + format(index, '05d') + '.png'
				seg_facade_name = seg_facade_dir + '/facade_' + format(index, '05d') + '.png'
				cv2.imwrite(real_facade_name, seg_img)
				cv2.imwrite(seg_facade_name, facade_img)
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

def main(aoi_dir, data):
	collect_segs(aoi_dir, data)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("aoi_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("data", help="path to input image folder (e.g., input_data)")
	args = parser.parse_args()

	main(args.aoi_dir, args.data)
