import os
import shutil

def main():
	"""
	Download the CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and 
	run this file to split the dataset into two domains, A and B. Further spilt into 
	trainA, trainB, testA, testB according to the split mentioned in the paper
	"""

	dataset_folder = '/data1/prane/CelebA/Img/img_align_celeba'
	attr_file_path = '/data1/prane/CelebA/Anno/list_attr_celeba.txt'

	# Define domain A = Male, B = Female

	# os.makedirs('trainA', exist_ok=True)
	# os.makedirs('trainB', exist_ok=True)
	# os.makedirs('testA', exist_ok=True)
	# os.makedirs('testB', exist_ok=True)

	# trainA_folder = './trainA'
	# trainB_folder = './trainB'
	# testA_folder = './testA'
	# testB_folder = './testB'

	os.makedirs('A', exist_ok=True)
	os.makedirs('B', exist_ok=True)

	A_folder = './A'
	B_folder = './B'

	# Open the attr file and identify the 'male' attribute
	attr_file = open(attr_file_path, 'r')
	image_attrs = attr_file.readlines()

	attr_def = image_attrs[1].split()
	male_index = attr_def.index('Male') + 1

	# Iterate over each labeled line
	for attr_line in image_attrs[2:]:
		attr_line = attr_line.split()
		image_id = attr_line[0]
		print(image_id)
		male_gender = attr_line[male_index]
		image_file = os.path.join(dataset_folder, image_id)

		if male_gender == '1':
			destination = A_folder
		else:
			destination = B_folder

		if os.path.isfile(image_file):
			shutil.copy(image_file, destination)
		else:
			print('Image id = ', image_id, ' does not exist')


if __name__ == '__main__':
	main()
