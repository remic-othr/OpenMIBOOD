import os
import zipfile
from tqdm import tqdm
import numpy as np
import nibabel as nib
from util import get_oasis_data, resample_volume


script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = f'{script_dir}/../../../data/oasis'
zip_path = f'{script_dir}/OASIS3.zip'

if os.path.exists(zip_path):
    zip_output = f'{script_dir}/tmp/OASIS3/'

    print('Extracting OASIS3 data. This may take a while ...')

    if not os.path.exists(os.path.join(zip_output, 'OASIS3', 'OAS31474')):
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            os.makedirs(zip_output, exist_ok=True)
            if not os.path.exists(os.path.join(zip_output, 'OAS31474')):
                zip_file.extractall(path=zip_output)
    # 'OASIS3/OAS30704/OAS30704_MR_d0584/anat3/NIFTI/sub-OAS30704_ses-d0584_run-02.nii.gz',
    # 'OASIS3/OAS30860/OAS30860_MR_d0638/anat4/NIFTI/sub-OAS30860_ses-d0638_run-02.nii.gz',
    # 'OASIS3/OAS30335/OAS30335_CT_d3820/CT1/NIFTI/sub-OAS30335_sess-d3820_CT.nii.gz',
    oasis3_data, oasis3_scanner_data, oasis3_ct_data = get_oasis_data()
        
    os.makedirs(base_path, exist_ok=True)
    
    # CT Preprocessing
    hu_min = 0
    hu_max = 80
    for path in (pbar := tqdm(oasis3_ct_data, desc='Processing CT scan:')):
        clipped_dir =  os.path.dirname(os.path.join(base_path, path)) # '../../../data/oasis/OASIS3/OAS30335/OAS30335_CT_d3820/CT1/NIFTI'
        path = os.path.join(zip_output, path) # tmp/OASIS3/OAS30335/OAS30335_CT_d3820/CT1/NIFTI/sub-OAS30335_sess-d3820_CT.nii.gz'
        assert(os.path.exists(path))
        path_dir = os.path.dirname(path) #  tmp/OASIS3/OAS30335/OAS30335_CT_d3820/CT1/NIFTI'
        path_filename = os.path.basename(path).split('.')[0] # 
        pbar.set_postfix({'Path': path_filename})

        clipped = f"{clipped_dir}/{path_filename}_clipped_{hu_min}_{hu_max}.nii.gz"
        if not os.path.exists(clipped):
            os.makedirs(clipped_dir, exist_ok=True)
            # Clip values between 0 and 80
            image = nib.load(path)
            image_np = image.get_fdata()
            image_np = np.clip(image_np, hu_min, hu_max)
            # Normalize to [0, 1]
            image_np = (image_np - hu_min) / (hu_max - hu_min)
            clipped_image = nib.Nifti1Image(image_np, affine=image.affine)
            nib.save(clipped_image, clipped)        
    print('OASIS-CT: Processing complete')
        
    # Scanner Preprocessing
    for path in (pbar := tqdm(oasis3_scanner_data, desc='Processing VISION device scans:')):
        scanner_dir =  os.path.dirname(os.path.join(base_path, path)) # '../../../data/oasis/OASIS3/OASIS3/OAS30860/OAS30860_MR_d0638/anat4/NIFTI'
        path_filename = os.path.basename(path).split('.')[0] # 
        pbar.set_postfix({'Path': path_filename})
        path = os.path.join(zip_output, path) # tmp/OASIS3/OASIS3/OAS30860/OAS30860_MR_d0638/anat4/NIFTI/sub-OAS30860_ses-d0638_run-02.nii.gz'
        
        assert(os.path.exists(path))
        image = nib.load(path)
        image_np = image.get_fdata()

        # The Vision device has a different orientation than the other scanners. This script corrects the orientation of the Vision MRI scans.
        image_np = np.transpose(image_np, (2, 1, 0))
        image_np = np.transpose(image_np, (0, 2, 1))
        image_np = np.flip(image_np, axis=1)
        image = nib.Nifti1Image(image_np, affine=image.affine)
        output_path = path.replace('.nii.gz', '_corrected.nii.gz')
        if not os.path.exists(output_path):
            nib.save(image, output_path)

        path_dir = os.path.dirname(output_path.replace('_corrected', ''))
        path_filename = os.path.basename(output_path.replace('_corrected', '')).split('.')[0]

        resampled_output = f"{path_dir}/{path_filename}_resampled.nii.gz"
        # Resample to isotropic 1mm^3
        if not os.path.exists(resampled_output):
            resample_volume(output_path, resampled_output, new_spacing=[1.0, 1.0, 1.0])

        skull_stripped_output = f"{scanner_dir}/{path_filename}_resampled_skull_stripped.nii.gz"
        os.makedirs(scanner_dir, exist_ok=True)
        if not os.path.exists(skull_stripped_output):
            command = f'hd-bet -i "{resampled_output}" -o "{skull_stripped_output}"'
            os.system(command)

    print('OASIS-Scanner: Processing complete')

    # MRI Preprocessing (T1w and T2w)
    for path in (pbar := tqdm(oasis3_data, desc='Processing T1w and T2w MRI scans:')):
        output_path = os.path.dirname(os.path.join(base_path, path))
        path = os.path.join(zip_output, path)
        path_dir = os.path.dirname(path)
        assert(os.path.exists(path))
        path_filename = os.path.basename(path).split('.')[0]
        pbar.set_postfix({'Path': path_filename})

        resampled_output = f"{path_dir}/{path_filename}_resampled.nii.gz"
        # Resample to isotropic 1mm^3
        if not os.path.exists(resampled_output):
            resample_volume(path, resampled_output, new_spacing=[1.0, 1.0, 1.0])

        skull_stripped_output = f"{output_path}/{path_filename}_resampled_skull_stripped.nii.gz"
        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(skull_stripped_output):
            command = f'hd-bet -i "{resampled_output}" -o "{skull_stripped_output}"'
            os.system(command)

else:
    print('To acquire the OASIS3 data, please request access at https://sites.wustl.edu/oasisbrains/home/access/. Only the access to OASIS-3: Longitudinal Multimodal Imaging is required.')
    print('Continue to register a NITRC account using the same email at https://www.nitrc.org/account/register.php.')
    print('After being granted access to the OASIS-3 data, log into your NITRC account and download the data.')
    print('This process is not as straightforward as the other datasets, so please follow the instructions on the github repository.')
    print('It comes down to selecting the following checkboxes: all sessions, T1w, T2w, and CT, download as zip archive, include project and subjects in file paths, simplify archive structure.')
    print('After downloading the data, rename the resulting file to "OASIS3.zip", place it in the directory of this script and rerun it.')
    print('Also, make sure that HD-BET (used for brain extraction) is installed (pip install hd-bet, to reproduce paper results, version 2.0.1 is required) and callable from the command line using "hd-bet".')