import os
import shutil
from PIL import Image
import cv2
import numpy as np

def create_domain_imgs(data_dir, dir_list):
	front = []
	back = []
	side = []

	# Iterate over every sub-category
	for sub_dir in dir_list:
		current_dir = os.path.join(data_dir, sub_dir)
		# Iterate over every image dir
		img_dir_list = os.listdir(current_dir)

		for image_dir in img_dir_list:
			if image_dir.startswith('id'):
				img_dir = os.path.join(current_dir, image_dir)
				# iterate over all images in img_dir
				img_files = os.listdir(img_dir)
				
				for img in img_files:
					img_path = os.path.join(img_dir, img)
					im = cv2.imread(img_path, cv2.IMREAD_COLOR)

					if 'front' in img:
						front.append(im)
					elif 'back' in img:
						back.append(im)
					elif 'side' in img:
						side.append(im)

	# Convert to numpy arrays
	front = np.array(front)
	back = np.array(back)
	side = np.array(side)

	print("Domain Images created")
	return front, back, side


def save_dataset(image_lists, domain):
	front, back, side = image_lists
	output_dir = './dataset/'
	output_path = os.path.join(output_dir, domain)

	np.savez(output_path + '.npz', front = front, back=back, side=side)
	print("Saved as npz file")


def main():
	data_dir = '/data1/prane/DeepFashion/In-shop_Clothes-Retrieval_Benchmark/Img/img/WOMEN'

	# Domain A -> Sleeveless or half sleeves tops
	# Domain B -> Jackets, Sweaters
	A_dir_list = ['Blouses_Shirts', 'Graphic_Tees', 'Tees_Tanks']
	B_dir_list = ['Cardigans', 'Jackets_Coats', 'Sweaters', 'Sweatshirts_Hoodies']

	# Get image lists
	A_image_lists = create_domain_imgs(data_dir, A_dir_list)
	B_image_lists = create_domain_imgs(data_dir, B_dir_list)

	# Save as datasets
	save_dataset(A_image_lists, 'A')
	save_dataset(B_image_lists, 'B')

if __name__ == '__main__':
	main()
