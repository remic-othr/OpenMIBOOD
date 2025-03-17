from util import get_response, download_with_curl, validate_patch
import zipfile
import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import time


print('##############################################################')
print('We will try to download the FNAC 2019 dataset automatically from https://1drv.ms/u/s!Al-T6d-_ENf6axsEbvhbEc2gUFs.')
download_path = 'tmp/fnac2019.zip'
print('If there are any errors with the automatic download, please visit the above link, download the file manually, move it to "tmp/fnac2019.zip", and run this script again.')
print('##############################################################')

root = '../../../data/midog/far/fnac2019_crops'
download_required = True

if os.path.exists(download_path):
    download_required = False

if download_required:
    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9',
        'appid': '1141147648',
        'cache-control': 'private',
        'content-type': 'application/json;odata=verbose',
        'origin': 'https://onedrive.live.com',
        'priority': 'u=1, i',
        'referer': 'https://onedrive.live.com/',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
        'x-forcecache': '1',
    }

    params = {
        'appId': '5cbed6ac-a083-4e14-b191-b4ba07653de2',
    }
    auth_token_url = 'https://api-badgerp.svc.ms/v1.0/token'
    response = get_response(auth_token_url, headers, params, type='post')
    authorization_badger = response['token']

    print(authorization_badger)
    time.sleep(2)

    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US',
        'authorization': f'Badger {authorization_badger}',
        'content-length': '0',
        'content-type': 'text/plain;charset=UTF-8',
        'origin': 'https://onedrive.live.com',
        'prefer': 'autoredeem',
        'priority': 'u=1, i',
        'referer': 'https://onedrive.live.com/',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
    }

    params = {
        '$select': 'id,parentReference',
    }
    url = 'https://my.microsoftpersonalcontent.com/_api/v2.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbC1UNmQtX0VOZjZheHNFYnZoYkVjMmdVRnM/driveitem'

    response = get_response(url, headers, params, type='post2')
    
    time.sleep(2)

    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US',
        'application': 'ODC Web',
        'authorization': f'Badger {authorization_badger}',
        'origin': 'https://onedrive.live.com',
        'priority': 'u=1, i',
        'referer': 'https://onedrive.live.com/',
        'scenario': 'DownloadFile',
        'scenariotype': 'AUO',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
    }

    params = {
        'select': 'id,@content.downloadUrl',
    }

    url = 'https://my.microsoftpersonalcontent.com/_api/v2.0/drives/FAD710BFDFE9935F/items/FAD710BFDFE9935F!107'

    response = get_response(url, headers, params)

    download_url = response['@content.downloadUrl'] # From https://1drv.ms/u/s!Al-T6d-_ENf6axsEbvhbEc2gUFs

    download_path = 'tmp/fnac2019.zip'
    download_with_curl(download_url, download_path)
    
if not os.path.exists(root) and os.path.exists(download_path):
    print('FNAC2019: Download complete')    
    with zipfile.ZipFile(download_path, 'r') as zip_file:
        zip_output = 'tmp/fnac2019'
        os.makedirs(zip_output, exist_ok=True)
        if not os.path.exists(os.path.join(zip_output, 'wg4bpm33hj-2')):
            zip_file.extractall(path=zip_output)

    subsets = ['B', 'M']


    morph_params = {'morph_kernel_open': 5, 'morph_kernel_erode': 3, 'morph_iter': 1}
    crop_nr = 0
    total_crop_nr = 0
    patch_size = 50
    patch_offset = 500
    keep = 10
    threshold = 100

    os.makedirs(root, exist_ok=True)

    for idx, subset in enumerate(subsets):
        crop_nr = 0
        prefix = f'{subset}_'
        input_dir = os.path.join(zip_output, subset)

        # Iterate over the files in the input directory
        for filename in tqdm(os.listdir(input_dir), desc=f'Processing images from subset {subset}'):
            # Extract file extension
            ext = os.path.splitext(filename)[1]
            
            # Open the image
            orig = Image.open(os.path.join(input_dir, filename))
            image = np.array(orig.convert('L')) # Convert to grayscale
            width = image.shape[1]
            height = image.shape[0]
            # Save in temp folder
            #cv2.imwrite('temp/00_image.png', image)
            _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
            # Invert the image
            thresh = cv2.bitwise_not(thresh)
            #cv2.imwrite('temp/01_thresh.png', thresh)
            #cv2.imwrite('temp/02_bitwise_not_thresh.png', thresh)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_params['morph_kernel_open'], morph_params['morph_kernel_open']))
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_params['morph_kernel_erode'], morph_params['morph_kernel_erode']))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=morph_params['morph_iter'])
            #cv2.imwrite('temp/03_open_thresh.png', thresh)
            thresh = cv2.erode(thresh, kernel_erode, iterations=morph_params['morph_iter'])
            #cv2.imwrite('temp/04_erode_thresh.png', thresh)
            info = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            labels = info[0]
            stats = info[2]
            # Sort stats by area
            stats = stats[stats[:, 4].argsort()[::-1]]
            # remove the first entry as it covers the whole image
            stats = stats[1:]
            # Only keep the top [keep] entries (at maximum) from the stats list
            stats = stats[:keep]

            # Calculate the centroid of each connected component
            centroids = []
            for i in range(len(stats)):
                x, y, w, h, area = stats[i]
                centroids.append((x+w//2, y+h//2))


            # # Create a directory for the image crops
            image_dir = os.path.join(root, f"{prefix}{os.path.splitext(filename)[0]}")
            os.makedirs(image_dir, exist_ok=True)
            # Create annotations dict
            annotations = []
            for i in range(len(stats)):
                x, y, w, h, area = stats[i]

                if area > 5:
                    center = centroids[i]
                    bbox = [center[0]-patch_size//2, center[1]-patch_size//2, center[0]+patch_size//2, center[1]+patch_size//2]
                    if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] < width and bbox[3] < height:
                        annotations.append({'bbox': bbox})
                        crop = orig.crop(bbox)
                        crop.save(os.path.join(image_dir, f"{total_crop_nr:05d}_cell{ext}"))
                        total_crop_nr += 1
                        crop_nr += 1

            
            # Create additional crops
            for annotation in annotations:
                bbox = annotation['bbox']
                additional_bbox = [bbox[0]+100, bbox[1], bbox[2]+100, bbox[3]]
                if validate_patch((width, height), annotations, additional_bbox):
                    crop = orig.crop(additional_bbox)
                    crop.save(os.path.join(image_dir, f"{total_crop_nr:05d}_additional{ext}"))
                    total_crop_nr += 1
                    crop_nr += 1
        

print('FNAC2019: Processing complete')