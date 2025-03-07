import os
import zipfile
from util import download_with_curl

zip_path = 'tmp/CATARACTS.zip'

zip_output = '../../../data/phakir/far/'
print('Downloading CATARACTS data...')
url = 'https://zenodo.org/records/14924735/files/CATARACTS.zip?download=1'
download_with_curl(url, zip_path)
if not os.path.exists(os.path.join(zip_output, 'CATARACTS')):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        os.makedirs(zip_output, exist_ok=True)
        zip_file.extractall(path=zip_output)

print('CATARACTS: Processing complete')


# import shutil
# import os
# import zipfile
# import tarfile
# from tqdm import tqdm

# base_path = '../../../data/phakir/far/CATARACTS/test'

# zip_path = 'images.zip'

# required_tars = ['test/micro_1.tar']

# if os.path.exists(zip_path):
#     zip_output = 'tmp/CATARACTS'
#     print('Extracting the first 5 test videos from CATARACTS. This may take a while...')
    
#     with zipfile.ZipFile(zip_path, 'r') as zip_file:
#         all_files = zip_file.namelist()
        
#         for file in all_files:
#             if file in required_tars:
#                 if not os.path.exists(os.path.join(zip_output, file)):
#                     print(f'Extracting {file}')
#                     zip_file.extract(file, path=zip_output)

#         os.makedirs(base_path, exist_ok=True)
#         for tar in required_tars:
#             tar_path = os.path.join(zip_output, tar)
#             assert(os.path.exists(tar_path))
#             with tarfile.open(tar_path, 'r') as tar_file:
#                 print(f'Extracting {tar} to {base_path}')
#                 tar_file.extractall(path=base_path)
    
    
#     print('CATARACTS: Processing complete')
# else:
#     print('To acquire the CATARACTS data, please register an account at https://ieee-dataport.org/open-access/cataracts.\nAfter signing in, download CATARACTS2018_images (691.2 GB) and place the resulting "images.zip" in the directory of this script and run this script again.')

