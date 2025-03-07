import os
import zipfile
from util import download_with_curl

zip_path = 'tmp/Cholec80_cropped.zip'

zip_output = '../../../data/phakir/near/'
print('Downloading Cholec80 data...')
url = 'https://zenodo.org/records/14921670/files/Cholec80_cropped.zip?download=1'
download_with_curl(url, zip_path)
if not os.path.exists(os.path.join(zip_output, 'Cholec80_cropped')):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        os.makedirs(zip_output, exist_ok=True)
        zip_file.extractall(path=zip_output)

print('Cholec80: Processing complete')
