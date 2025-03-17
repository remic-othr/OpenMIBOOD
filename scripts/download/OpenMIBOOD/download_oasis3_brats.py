import os
import zipfile
from tqdm import tqdm
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = f'{script_dir}/../../../data/oasis/near/BraTS2023-GLI'

zip_path = f'{script_dir}/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip'

if os.path.exists(zip_path):
    zip_output = f'{script_dir}/tmp/BraTS2023-GLI'
    print('Extracting BraTS2023 data. This may take a while...')
    
    if not os.path.exists(os.path.join(base_path, 'BraTS-GLI-01666-000')):
        with zipfile.ZipFile(zip_path, 'r') as zip_file:

            if not os.path.exists(os.path.join(zip_output, f'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-01666-000')):
                os.makedirs(zip_output, exist_ok=True)
                zip_file.extractall(path=zip_output)
            
        os.makedirs(base_path, exist_ok=True)
        source_path = os.path.join(zip_output, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')

        for item in tqdm(os.listdir(source_path), desc='Moving files'):
            file_list = os.listdir(os.path.join(source_path, item))
            for file in file_list:
                if 't1n' in file: # T1-weighted MRI
                    os.makedirs(os.path.join(base_path, item), exist_ok=True)
                    shutil.move(os.path.join(source_path, item, file), os.path.join(base_path, item))
                    break
    
    print('BraTS2023: Processing complete')
else:
    print('To acquire the BraTS 2023 data, please register at https://www.synapse.org/Synapse:syn51156910/wiki/627000 and fill out the data access form.')
    print('After joining the data access team, download the "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" file, place it in this script\'s directory, and rerun the script.')

