from util import get_response, download_with_curl, validate_patch
import zipfile
import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2


morph_params = {'morph_kernel_open': 5, 'morph_kernel_erode': 3, 'morph_iter': 1}

root = '../../../data/midog/far/fnac2019_crops'

if not os.path.exists(root):
    headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US',
            'Referer': 'https://onedrive.live.com/',
            'Scenario': 'DownloadFile',
            'ScenarioType': 'AUO',
            'Application': 'ODC Web',
            'Origin': 'https://onedrive.live.com',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Authorization': 'Badger eyJhbGciOiJSUzI1NiIsImtpZCI6IjEzQTAwRkQ1MEEzMEM1MTVDQjYzMDNFREI3NEE2MTlBNzQ0NUQzRkEiLCJ4NXQiOiJFNkFQMVFvd3hSWExZd1B0dDBwaG1uUkYwX28iLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJodHRwczovL29uZWRyaXZlLmNvbS8iLCJpc3MiOiJodHRwczovL2JhZGdlci5zdmMubXMvdjEuMC9hdXRoIiwiZXhwIjoxNzQwNDg2ODAyLCJuYmYiOjE3Mzk4ODIwMDIsImdpdmVuX25hbWUiOiI5NyIsImZhbWlseV9uYW1lIjoiUmhpbm9jZXJvcyIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL3NpZCI6IjhjMzNlNzBjMzMzNTFiOTI0MTFkNGRlMTdlZWIyMjNkIiwiYXBwaWQiOiI1Y2JlZDZhYy1hMDgzLTRlMTQtYjE5MS1iNGJhMDc2NTNkZTIiLCJpYXQiOjE3Mzk4ODIwMDJ9.SGQdUBRQjFdeerJCjih5DCNZixwHZ1aomvzssnMo9ZFKkPi4VV7WjEIwOxXD8CKuW_wUyffoP3A_f01FQBcirrhPFfYEsYWBOBiKTg1MNZguSGY0KgNYrQB19KA1EUpLWbArnrvVhtwoTHggAEBWliWDKvINm-UNGueTYLjB-HvMh6LVsgRXlVGkkV-pO41Ppac5Rv5hH9s1myZED0tRiEoneCB05gWR6kqsNdjHWP1tLFlN_wrzR55sEqXS43umKPTiOPTTQcsSm_skmlUsUa00qXqmyOXvo1_A5heu3Ny7yx7W0fkouJ6-3n7HLv_oTdUDbwBW_dhN8-Z0I1aC4g',
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache'
        }

    params = {
        'select': 'id,@content.downloadUrl',
    }

    url = 'https://my.microsoftpersonalcontent.com/_api/v2.0/drives/FAD710BFDFE9935F/items/FAD710BFDFE9935F!107'
    response = get_response(url, headers, params)
    download_url = response['@content.downloadUrl'] # From https://1drv.ms/u/s!Al-T6d-_ENf6axsEbvhbEc2gUFs

    download_path = 'tmp/fnac2019.zip'
    download_with_curl(download_url, download_path)
    with zipfile.ZipFile(download_path, 'r') as zip_file:
        zip_output = 'tmp/fnac2019'
        os.makedirs(zip_output, exist_ok=True)
        if not os.path.exists(os.path.join(zip_output, 'wg4bpm33hj-2')):
            zip_file.extractall(path=zip_output)

    subsets = ['B', 'M']

    print('FNAC2019: Download complete')    

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