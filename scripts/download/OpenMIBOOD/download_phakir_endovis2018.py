import shutil
import os
import zipfile
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
zip_paths = [f'{script_dir}/Endovis2018_1.zip', f'{script_dir}/Endovis2018_2.zip']

base_path = f'{script_dir}/../../../data/phakir/near/Endovis2018'

if os.path.exists(zip_paths[0]) and os.path.exists(zip_paths[1]):
    zip_output = f'{script_dir}/tmp/Endovis2018'
    for zip_path in tqdm(zip_paths, desc='Extracting Endovis2018 data'):
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            os.makedirs(zip_output, exist_ok=True)
            zip_file.extractall(path=zip_output)
    
    sequences = ['seq_1', 'seq_2', 'seq_3', 'seq_4']
    for sequence in tqdm(sequences, desc='Copying Endovis2018 data'):
        output_path = os.path.join(base_path, sequence)
        input_path = os.path.join(zip_output, 'test_data', sequence, 'left_frames')

        shutil.copytree(input_path, output_path)
    
    print('Endovis2018: Processing complete')
else:                                                                    
    print('To acquire the Endovis2018 data, please register an account at https://endovissub2018-roboticscenesegmentation.grand-challenge.org/.')
    print('After joining the challenge, the data can be downloaded.\nPlease make sure that you only download the test dataset consisting of 4 sequences.')
    print('After the download is complete, rename the two files to Endovis2018_1.zip and Endovis2018_2.zip, move them to this directory and run this script again.')

