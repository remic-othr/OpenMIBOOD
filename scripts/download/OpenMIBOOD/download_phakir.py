import os
import zipfile
import os
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = f'{script_dir}/../../../data/phakir'
zip_path = f'{script_dir}/phakir_openmibood.zip'
zip_output = f'{script_dir}/../../../data/phakir/'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        os.makedirs(zip_output, exist_ok=True)
        if not os.path.exists(os.path.join(zip_output, 'Video_07')):
            zip_file.extractall(path=zip_output)

    print('PhaKIR: Processing complete')
else:
    print('To access the PhaKIR dataset, visit https://doi.org/10.5281/zenodo.16753918 and follow the instructions to receive access.')
    print('After acquiring the dataset, place the phakir_openmibood.zip in the directory of this script and run this script again..')

