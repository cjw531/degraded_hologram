from tqdm import tqdm
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Mask out the area outside of ROI (hologram)
Need to have a pre-defined pixel coordinates
Pre-defined 4 coordinates can be in a parallelogram-shape
"""

def mask_image_smooth(is_grayscale: bool, image_path: str, corners: list, border_width=300, debug: bool=False):
    """
    Masks out the region outside the given quadrilateral corners and creates a smooth border.
    @param image_path: Path to the input image.
    @param corners: List of four (x, y) coordinates defining the quadrilateral.
    @param border_width: Width of the smoothing border in pixels.
    """

    if is_grayscale: image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else: image = cv2.imread(image_path) # read image

    if image is None:
        raise ValueError("Image not found or invalid path.")
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8) # empty mask, same shape as an image
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2)) # corners into numpy array, reshape for polygon fill
    cv2.fillPoly(mask, [pts], 255) # polygon fill
    mask_img = cv2.bitwise_and(image, image, mask=mask) # masked image

    border_mask = np.zeros(image.shape[:2], dtype=np.uint8) # extended border area (smooth out region)mask_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    x_coords = [p[0] for p in corners] # x-coord only
    y_coords = [p[1] for p in corners] # y-coord only
    min_x, max_x = min(x_coords), max(x_coords) # min & max coords among x-coords
    min_y, max_y = min(y_coords), max(y_coords) # min & max coords among y-coords
    
    # border area coordinates
    min_x_border = max(0, min_x - border_width)
    min_y_border = max(0, min_y - border_width)
    max_x_border = min(image.shape[1], max_x + border_width)
    max_y_border = min(image.shape[0], max_y + border_width)
    
    border_pts = np.array([[min_x_border, min_y_border],
        [max_x_border, min_y_border],
        [max_x_border, max_y_border],
        [min_x_border, max_y_border]
    ], dtype=np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(border_mask, [border_pts], 255) # polygon fill

    border_pixels = image.copy()
    border_pixels = cv2.bitwise_and(border_pixels, border_pixels, mask=255 - mask) # exclude the masked area from the border pixels
    border_pixels = cv2.bitwise_and(border_pixels, border_pixels, mask=border_mask) # include only the border area in the border pixels

    if is_grayscale:
        border_pixel_values = border_pixels[border_pixels > 0]
    else:
        border_pixel_values = border_pixels[border_pixels.any(axis=2)]
    # border_pixel_values = mask_img[mask_img.any(axis=2)] # little brighter pixel values

    # average pixel value based on the border of the hologram
    if border_pixel_values.size > 0: avg_border_color = np.mean(border_pixel_values, axis=0).astype(np.uint8)
    else: avg_border_color = np.array(0 if is_grayscale else [0, 0, 0], dtype=np.uint8)

    if is_grayscale:
        color_scheme = np.linspace(avg_border_color, 0, border_width + 1, dtype=np.uint8)[1:]
    else:
        color_scheme = np.linspace(avg_border_color, [0, 0, 0], border_width + 1, dtype=np.uint8)[1:]
    smooth_masked_image = image.copy()
    layer_masks = [mask.copy()] # original polygon mask

    for i in range(border_width): # dilate the mask to create layers from the border
        dilated_mask = cv2.dilate(layer_masks[-1], np.ones((3, 3), np.uint8), iterations=1)
        layer_masks.append(dilated_mask)

    for i in range(border_width):
        color = color_scheme[i] # current layer's color
        layer_mask = layer_masks[i + 1] - layer_masks[i] # current layer mask
        if is_grayscale:
            smooth_masked_image = np.where(layer_mask == 255, color, smooth_masked_image).astype(np.uint8)
        else:
            smooth_masked_image = np.where(layer_mask[:, :, np.newaxis] == 255, color, smooth_masked_image).astype(np.uint8)

    # fill the gap with black
    gap_mask = border_mask - layer_masks[-1]
    if is_grayscale:
        smooth_masked_image = np.where(gap_mask == 255, 0, smooth_masked_image).astype(np.uint8)
    else:
        smooth_masked_image = np.where(gap_mask[:, :, np.newaxis] == 255, [0, 0, 0], smooth_masked_image).astype(np.uint8)

    # combine original image & smooth mask
    if is_grayscale:
        smooth_masked_image = np.where(mask == 255, image, smooth_masked_image).astype(np.uint8)
    else:
        smooth_masked_image = np.where(mask[:, :, np.newaxis] == 255, image, smooth_masked_image).astype(np.uint8)
    smooth_masked_image = cv2.bitwise_and(smooth_masked_image, smooth_masked_image, mask=border_mask) # apply border mask to the smooth mask image

    if debug: # if debug mode on, plot the output
        debug_plot(image, mask_img, smooth_masked_image)

    return mask_img, smooth_masked_image

def debug_plot(original, mask, smooth_mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    axes[1].imshow(mask)
    axes[1].axis("off")
    axes[1].set_title("Masked Image")

    axes[2].imshow(smooth_mask)
    axes[2].axis("off")
    axes[2].set_title("Smoothed Border Image")

    plt.show()

def read_filename_coordinates(grayscale: bool, undistorted: bool, csv_file_path: str="coordinates.csv"):
    filename, coordinates = [], []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        for row in reader:
            filename.append(row[0])
            coordinates.append([str_to_tuple(row[1]), str_to_tuple(row[2]), str_to_tuple(row[3]), str_to_tuple(row[4])])

    if grayscale: # adjust postfix of the filename, drop RGB at the end of the filename
        filename = [remove_rgb_postfix(f) for f in filename]

    if undistorted:
        filename = [filename_undistorted(f) for f in filename]
    
    return filename, coordinates

def str_to_tuple(tuple_str: str):
    """
    Convert string to tuple
    @param tuple_str: tuple in string format
    @return: tuple
    """

    numbers = tuple_str.strip('()').split(', ')
    return tuple(map(int, numbers))

def remove_rgb_postfix(filename):
    # TODO: hardcoded
    filename = filename.replace(" - RGB", "")
    return filename

def filename_undistorted(filename):
    # TODO: hardcoded
    filename = filename.replace(".png", "")
    filename += "UNDISTORTED.png"
    return filename

if __name__ == '__main__':
    base_path = "../grayscale_calib_merged/images/"
    filename, coordinates = read_filename_coordinates(grayscale=True, undistorted=True, csv_file_path="coordinates.csv")

    for file, coord in tqdm(zip(filename, coordinates), total=len(filename)):
        mask_img, smooth_masked_image = mask_image_smooth(is_grayscale=True, image_path=os.path.join(base_path, file), corners=coord)
        cv2.imwrite(os.path.join("../grayscale_calib_merged_mask/images/", file), mask_img)
        cv2.imwrite(os.path.join("../grayscale_calib_merged_mask_smooth/images/", file), smooth_masked_image)
