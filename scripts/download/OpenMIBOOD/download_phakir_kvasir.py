from util import download_with_curl
import zipfile
import os

import shutil

output_base_path = '../../../data/phakir/far/kvasir-seg'

if not os.path.exists(os.path.join(output_base_path)):
    url = 'https://datasets.simula.no/downloads/kvasir-seg.zip'
    download_path = 'tmp/kvasir-seg.zip'
    download_with_curl(url, download_path)
    with zipfile.ZipFile(download_path, 'r') as zip_file:
        zip_output = 'tmp/kvasir-seg'
        os.makedirs(zip_output, exist_ok=True)
        zip_file.extractall(path=zip_output)

    input_path = os.path.join(zip_output, 'Kvasir-SEG', 'images')

    shutil.copytree(input_path, output_base_path)


print('Kvasir-Seg: Processing complete')