from util import download_with_curl, validate_patch
import zipfile
import os
from tqdm import tqdm
import cv2
import numpy as np

def get_largest_contour(image):
    # Find contours
    min_contour_area = 5000  # Set a minimum contour area to ignore small blobs
    min_circularity = 0.995   # Set a minimum circularity threshold
    max_ratio = 1.05
    min_ratio = 0.95
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_contour = None
    circularity = 0
    largest_contour_ratio = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if len(cnt) > 5 and area > min_contour_area:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, _) = ellipse
            
            ellipse_center = (int(center[0]), int(center[1]))

            # Check if the center is roughly in the center of the image
            if ellipse_center[0] < 0.40 * image.shape[1] or ellipse_center[1] > 0.58 * image.shape[0]:
                continue
            if ellipse_center[0] > 0.60 * image.shape[1] or ellipse_center[1] < 0.42 * image.shape[0]:
                continue

            ellipse_width, ellipse_height = axes
            ratio = ellipse_height / ellipse_width
            # Calculate semi-major and semi-minor axes
            a = ellipse_height / 2  # semi-major axis
            b = ellipse_width / 2  # semi-minor axis

            # Calculate the area of the ellipse
            area = np.pi * a * b

            # Calculate the perimeter of the ellipse using Ramanujan's approximation
            perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
                # Filter based on both circularity
                if circularity >= min_circularity and ratio <= max_ratio and ratio >= min_ratio:
                    if area > largest_area:
                        largest_area = area
                        largest_contour = cnt
                        largest_contour_ratio = ellipse_height / ellipse_width

    return largest_contour, largest_area, circularity, contours, largest_contour_ratio

script_dir = os.path.dirname(os.path.abspath(__file__))
output_base_path = f'{script_dir}/../../../data/phakir/near/Endovis2015_cropped'

if not os.path.exists(os.path.join(output_base_path, 'OP4/Raw/')):
    url = 'https://opencas.webarchiv.kit.edu/data/endovis15_ins/Segmentation_Rigid_Training.zip'
    download_path = f'{script_dir}/tmp/Endovis2015.zip'
    download_with_curl(url, download_path)
    with zipfile.ZipFile(download_path, 'r') as zip_file:
        zip_output = f'{script_dir}/tmp/Endovis2015'
        os.makedirs(zip_output, exist_ok=True)
        if not os.path.exists(os.path.join(zip_output, 'Segmentation_Rigid_Training/Training/OP4')):
            zip_file.extractall(path=zip_output)

base_path = f'{script_dir}/tmp/Endovis2015/Segmentation_Rigid_Training/Training/'

surgery_list = [f'OP{idx}/Raw/' for idx in range(1, 5)]

vignette_surgeries = [f'OP2/Raw/', 'OP4/Raw/']
rectangle_surgeries = ['OP1/Raw/', 'OP3/Raw/']

for idx, surgery in enumerate(tqdm(surgery_list, desc='Processing surgeries')):            
    surgery_path = os.path.join(base_path, surgery)   
    output_path = os.path.join(output_base_path, surgery)
    os.makedirs(output_path, exist_ok=True)

    for frame_name in tqdm(os.listdir(surgery_path), desc='Processing frames', leave=False):
        frame_path = os.path.join(surgery_path, frame_name)
        frame = cv2.imread(frame_path)
        if surgery in rectangle_surgeries:
            if idx == 0:
                y_top_offset = 8
                x_left_offset = 4
                y_bottom_offset = 471
                x_right_offset = 634
            if idx == 2:
                y_top_offset = 0
                x_left_offset = 0
                y_bottom_offset = 477
                x_right_offset = 640
            frame = frame[y_top_offset:y_bottom_offset, x_left_offset:x_right_offset]
        elif surgery in vignette_surgeries:            
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            value_channel = hsv_image[:, :, 2]  # V channel

            # Iterate over a range of threshold values to find the best one
            best_circularity = 0
            best_threshold = 0
            best_contour = None
            for threshold_value in range(3, 35):
                _, frame_thresh = cv2.threshold(value_channel, threshold_value, 255, cv2.THRESH_BINARY)

                kernel = np.ones((5, 5), np.uint8)
                morph_image = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)
                morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)

                largest_contour, largest_area, circularity, contours, largest_contour_ratio = get_largest_contour(morph_image) 
                if largest_contour is None:
                    continue

                if circularity > best_circularity:
                    best_circularity = circularity
                    best_threshold = threshold_value
                    best_contour = largest_contour
                    best_contour_ratio = largest_contour_ratio

            if best_contour is None:
                print(f'No contour found for frame {frame_name}')
                continue
            else:
                pass
                
            ellipse = cv2.fitEllipse(best_contour)
                
            # Calculate the maximum fitting rectangle with aspect ratio 16:9 positioned in the center of the frame
            assert(ellipse is not None)

            (center, axes, angle) = ellipse
            ellipse_center = (int(center[0]), int(center[1]))
            ellipse_width, ellipse_height = axes
            ellipse_axes = (int(ellipse_width / 2), int(ellipse_height / 2))  # Half the axes to get the radius

            # Ensure the rectangle stays within the calculated ellipse
            rect_center_x = ellipse_center[0]
            rect_center_y = ellipse_center[1]

            # Calculate the maximum height of the rectangle
            max_height = ellipse_axes[1] / (16. / 9.)

            tan = max_height / ellipse_axes[0]
            arc_tan = np.arctan(tan)

            # * 0.97 to shrink the rectangle by 3% and increase the likelihood it stays within the ellipse over all frames
            rect_width = int((np.cos(arc_tan) * axes[0])* 0.97)
            rect_height = int(rect_width / (16. / 9.))

            x_offset = (rect_center_x - rect_width // 2)
            y_offset = (rect_center_y - rect_height // 2)

            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)
            x_offset = min(frame.shape[1] - rect_width, x_offset)
            y_offset = min(frame.shape[0] - rect_height, y_offset)

            frame = frame[y_offset:y_offset + rect_height, x_offset:x_offset + rect_width]
        cv2.imwrite(os.path.join(output_path, frame_name), frame)

print('EndoVis2015: Processing complete')