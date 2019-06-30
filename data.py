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


def main(aoi_dir, html_file_name):

	# Create the html file
	html_file = "<html>\n"
	html_file += "  <head>\n"
	html_file += "    <style>\n"
	html_file += "    table {\n"
	html_file += "      color: #333;\n"
	html_file += "      font-family: Helvetica, Arial, sans-serif;\n"
	html_file += "      border-collapse: collapse;\n"
	html_file += "      border-spacing: 0;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    td, th {\n"
	html_file += "      border: 1px solid #CCC;\n"
	html_file += "      padding: 5px;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    th {\n"
	html_file += "      background: #F3F3F3;\n"
	html_file += "      font-weight: bold;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    td {\n"
	html_file += "      text-align: center;\n"
	html_file += "    }\n"
	html_file += "    \n"
	html_file += "    tr:hover {\n"
	html_file += "      background-color: #eef;\n"
	html_file += "      border: 3px solid #00f;\n"
	html_file += "      font-weight: bold;\n"
	html_file += "    }\n"
	html_file += "    </style>\n"
	html_file += "  </head>\n"
	html_file += "<body>\n"
	html_file += "  <table>\n"
	html_file += "    <tr>\n"
	html_file += "      <th>Cluster.</th>\n"
	html_file += "      <th>Facade.</th>\n"
	html_file += "      <th>Facade Seg.</th>\n"
	aoi_areas = sorted(os.listdir(aoi_dir))
	for l in range(len(aoi_areas)):
		clusters = sorted(os.listdir(aoi_dir + '/' + aoi_areas[l]))
		for i in range(len(clusters)):
			cluster = aoi_dir + '/' + aoi_areas[l] + '/' + clusters[i]
			if not os.path.exists(cluster + '/seg'):
				continue
			seg_images = sorted(os.listdir(cluster + '/seg'))
			for j in range(len(seg_images)):
				facade_img_name = cluster + '/image/' + seg_images[j]
				seg_img_name = cluster + '/seg/' + seg_images[j]
				seg_img = cv2.imread(seg_img, cv2.IMREAD_UNCHANGED)
				# find the rectangle

				html_file += "    <tr>\n"
				html_file += "      <td>" + aoi_areas[l] + '_' + clusters[i] + '_' + seg_images[j][: len(seg_images[j]) - 4] + "</td>\n"
				html_file += "      <td><a href=\"" + facade_img_name + "\"><img src=\"" + facade_img_name + "\"/></a></td>\n"
				html_file += "      <td><a href=\"" + seg_img_name + "\"><img src=\"" + seg_img_name + "\"/></a></td>\n"
				html_file += "    </tr>\n"

	html_file += "  </table>\n"
	html_file += "</body>\n"
	html_file += "</html>\n"
		
	# Save the html file
	with open(html_file_name, "w") as output_file:
		output_file.write(html_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("aoi_dir", help="path to input image folder (e.g., input_data)")
	parser.add_argument("html_file_name", help="path to output html filename")
	args = parser.parse_args()

	main(args.aoi_dir, args.html_file_name)
