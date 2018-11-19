import os
import shutil
from PIL import Image
import cv2
import numpy as np


def create_dataset(domain_file_list):
	file_paths_dir = os.path.join(os.getcwd(), 'image_paths')
	all_images = []

	# Iterate over each file in domain_file_list
	for domain_file in domain_file_list:
		file_path = os.path.join(file_paths_dir, domain_file)
		all_paths = open(file_path, 'r').readlines()

		# Iterate over each path
		for path in all_paths:
			img_path = path.split()[0]
			im = cv2.imread(img_path, cv2.IMREAD_COLOR)
			all_images.append(im)

	all_images = np.array(all_images)

	print('Dataset created')
	return all_images


def save_dataset(all_images, domain):
	dataset_dir = './dataset/'
	output_path = os.path.join(dataset_dir, domain + '_path.npy')

	np.save(output_path, all_images)
	print('Saved!')


def main():
	domain_A = ['female_tanks.txt']
	domain_B = ['male_tanks.txt']

	A_image_list = create_dataset(domain_A)
	B_image_list = create_dataset(domain_B)

	save_dataset(A_image_list, 'A')
	save_dataset(B_image_list, 'B')



if __name__ == '__main__':
	main()
