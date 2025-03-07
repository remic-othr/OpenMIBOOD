import os
import tarfile
from tqdm import tqdm
import shutil
from util import download_with_gdown

base_path = '../../../data/oasis/far/SegmentationDecathlon'
tar_path = 'tmp/Task02_Heart.tar'

tar_output = f'tmp/SegmentationDecathlon/Task02_Heart'
print('Downloading Heart data...')
id = '1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY'
download_with_gdown(id, tar_path)
if not os.path.exists(os.path.join(base_path, 'Task02_Heart/imagesTr')):
    with tarfile.open(tar_path, 'r') as tar_file:
        if not os.path.exists(os.path.join(tar_output, 'imagesTr')):
            tar_file.extractall(path=tar_output)
        
    os.makedirs(base_path, exist_ok=True)
    shutil.move(os.path.join(tar_output, 'Task02_Heart'), base_path)
    
print('Heart: Processing complete')
