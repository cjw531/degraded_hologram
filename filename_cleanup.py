import os
import re
import argparse
from tqdm import tqdm


def rename_files_in_folder(folder_path):
    files = os.listdir(folder_path) # list all files in the folder
    
    # Loop through each file in the folder
    for filename in files:
        new_filename = filename

        # Replace spaces with no space
        new_filename = new_filename.replace(' ', '')

        # Replace commas with underscores
        new_filename = new_filename.replace(',', '_')

        # Replace - with _ unless it's between a letter and a digit, in which case replace with _neg_
        new_filename = re.sub(r'([A-Za-z])-(\d)', r'\1_neg_\2', new_filename)
        new_filename = new_filename.replace('-', '_')

        # Replace degrees with degree and add underscore after degree
        new_filename = new_filename.replace('degrees', 'degree')
        new_filename = new_filename.replace('degree', '_degree')

        new_filename = new_filename.replace('UNDISTORTED', '_calib') # "UNDISTORTED" from JT calibration

        new_filename = re.sub(r'(A)', r'\1_', new_filename) # Add underscore after "A"

        # Replace multiple consecutive underscores with a single underscore
        new_filename = re.sub(r'_+', '_', new_filename)

        # construct the full file paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} ->\t {new_filename}')

if __name__ == "__main__":
    # Define the folder path containing the images
    folder_path = '../grayscale_calib_merged/images/'
    print(os.listdir(folder_path))
    # rename_files_in_folder(folder_path)