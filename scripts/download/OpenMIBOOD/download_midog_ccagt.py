from util import download_with_curl, validate_patch
import zipfile
import os
import json
from PIL import Image
from tqdm import tqdm

directories = ['wg4bpm33hj-2/images/A.zip', 'wg4bpm33hj-2/images/B.zip', 'wg4bpm33hj-2/images/C.zip', 'wg4bpm33hj-2/images/D.zip', 'wg4bpm33hj-2/images/E.zip', 
                         'wg4bpm33hj-2/images/F.zip', 'wg4bpm33hj-2/images/G.zip', 'wg4bpm33hj-2/images/H.zip', 'wg4bpm33hj-2/images/I.zip', 'wg4bpm33hj-2/images/J.zip', 
                         'wg4bpm33hj-2/images/K.zip', 'wg4bpm33hj-2/images/L.zip', 'wg4bpm33hj-2/images/M.zip', 'wg4bpm33hj-2/images/N.zip', 'wg4bpm33hj-2/images/O.zip']
script_dir = os.path.dirname(os.path.abspath(__file__))
root = f'{script_dir}/../../../data/midog/far/ccagt_crops'

if not os.path.exists(root):
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/wg4bpm33hj-2.zip'
    download_path = f'{script_dir}/tmp/ccagt.zip'
    download_with_curl(url, download_path)
    with zipfile.ZipFile(download_path, 'r') as zip_file:
        zip_output = f'{script_dir}/tmp/ccagt'
        os.makedirs(zip_output, exist_ok=True)
        if not os.path.exists(os.path.join(zip_output, 'wg4bpm33hj-2')):
            zip_file.extractall(path=zip_output)

    base_path = os.path.join(zip_output, 'subsets')
    os.makedirs(base_path, exist_ok=True)
    for directory in tqdm(directories, desc='Extracting CCAgT subsets...'):
        source = directory
        source_path = os.path.join(zip_output, source)
        filename = os.path.splitext(os.path.basename(source_path))[0]
        if not os.path.exists(os.path.join(base_path, filename)):
            # Handling nested Zip-Files
            with zipfile.ZipFile(source_path, 'r') as inner_zip:
                inner_zip.extractall(path=base_path)

    print('CCAgT: Download complete')    
    ccagt_data = json.load(open(os.path.join(zip_output,'wg4bpm33hj-2', 'CCAgT_COCO_OD.json')))

    test_split = ['E', 'K', 'L']
    valid_split = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'M', 'N', 'O']

    preprocessed_annotations = []
    width = 1600
    height = 1200
    for idx, annotation in enumerate(tqdm(ccagt_data['annotations'], desc='Preprocessing CCAgT annotations...')):
        # Transform annotations to bbox format
        annotation['bbox'] = [int(coord) for coord in annotation['bbox']]
        # take the center of the bbox\
        annotation['bbox'] = [annotation['bbox'][0] + annotation['bbox'][2]//2, annotation['bbox'][1] + annotation['bbox'][3]//2]
        # create a 50x50 bbox around the center
        annotation['bbox'] = [annotation['bbox'][0]-25, annotation['bbox'][1]-25, annotation['bbox'][0]+25, annotation['bbox'][1]+25]

        if annotation['category_id'] == 2 or annotation['category_id'] == 3:
            continue # Cluster and Satellite are inside nuclei, so we ignore them
        if annotation['bbox'][0] < 0 or annotation['bbox'][1] < 0 or annotation['bbox'][2] >= width or annotation['bbox'][3] >= height:
            continue
        preprocessed_annotations.append(annotation)

    for annotation in tqdm(preprocessed_annotations, desc='Processing CCAgT annotations...'):
        bbox = annotation['bbox']
        annotation_id = annotation['id']
        image_id = annotation['image_id']
        image = [img for img in ccagt_data['images'] if img['id'] == image_id][0]
        file_name = image['file_name']
        slide_id = file_name.split('_')[0]
        target = 'test' if slide_id in test_split else 'valid'

        # Create the output directory if it does not exist
        out_dir = os.path.join(root, file_name)
        os.makedirs(out_dir, exist_ok=True)
        image = Image.open(os.path.join(base_path, slide_id, f'{file_name}.jpg'))
        crop = image.crop(bbox)
        crop.save(os.path.join(out_dir, f"{annotation_id}_{annotation['category_id']}.jpg"))

        # Create additional crops
        additional_bbox = [bbox[0]+100, bbox[1], bbox[2]+100, bbox[3]]

        annotations_by_img_id = [ann for ann in preprocessed_annotations if ann['image_id'] == image_id]
        # Transform annotations to bbox format

        if validate_patch((width, height), annotations_by_img_id, additional_bbox):
            crop = image.crop(additional_bbox)
            crop.save(os.path.join(out_dir, f"{annotation_id}_{annotation['category_id']}_additional.jpg"))
        

print('CCAgT: Processing complete')