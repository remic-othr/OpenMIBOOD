import os
import zipfile
from tqdm import tqdm
import shutil
from util import download_with_curl, dicom2nifti
import glob


# Function to extract the numerical part from the filename
def extract_number(file_path):
    number = file_path.split('-')[-1]
    if number.isdigit() and (int(number) < 1000):
        return int(number)
    
    raise ValueError(f"Could not extract number from {file_path}")

# Function to extract the numerical part from the filename
def extract_number(file_path):
    number = file_path.split('-')[-1].split('.')[0]
    if number.isdigit() and (int(number) < 1000):
        return int(number)
    
    raise ValueError(f"Could not extract number from {file_path}")


base_path = '../../../data/oasis/far/CHAOS/NIFTI'
zip_path = 'tmp/CHAOS_Test_Sets.zip'

zip_output = f'tmp/CHAOS'
print('Downloading CHAOS data...')
url = 'https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip?download=1'
download_with_curl(url, zip_path)
if not os.path.exists(os.path.join(base_path, 'NIFTI')):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        os.makedirs(zip_output, exist_ok=True)
        if not os.path.exists(os.path.join(zip_output, 'Test_Sets')):
            zip_file.extractall(path=zip_output)

    CHAOS_path = f'{zip_output}/Test_Sets/MR'
    for subject in tqdm(os.listdir(CHAOS_path)):
        subject_path = os.path.join(CHAOS_path, subject)
        subject_path = os.path.join(subject_path, 'T1DUAL', 'DICOM_anon')

        phases = ['InPhase']
        for phase in phases:
            phase_output = os.path.join(base_path, phase)
            os.makedirs(phase_output, exist_ok=True)
            phase_path = os.path.join(subject_path, phase)

            files = glob.glob(os.path.join(f"{phase_path}/*.dcm"))
            files = sorted(files, key=extract_number)
            nifti_output = f"{phase_output}/{subject}.nii.gz"
            try:
                dicom2nifti(files, nifti_output)
            except Exception as e:
                print(f'Could not convert {subject} to NIfTI')
                continue
    
print('CHAOS: Processing complete')
