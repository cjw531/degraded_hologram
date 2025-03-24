import os
import argparse
import cv2
from tqdm import tqdm

def tiff_to_png(tiff_file_path: str):
    tiff_image = cv2.imread(tiff_file_path, cv2.IMREAD_UNCHANGED)

    if tiff_image is None:
        raise ValueError("Image not found or invalid path.")

    cv2.imwrite(tiff_file_path.rsplit('.', 1)[0] + '.png', tiff_image) # write into png file
    os.remove(tiff_file_path) # remove the tiff file (overwrite)


if __name__ == '__main__':
    '''
    Argparse Instruction:
    $ python tiff_to_png.py "../grayscale_merged/images/"
    '''

    parser = argparse.ArgumentParser(description='Convert TIFF images to PNG and delete the original TIFF files')
    parser.add_argument('folder_path', type=str, help='image folder with the *.tiff images')
    
    args = parser.parse_args()
    folder_path = args.folder_path
    
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for f in tqdm(file_list):
        tiff_to_png(os.path.join(folder_path, f))
        