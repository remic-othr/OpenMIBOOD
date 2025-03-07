import os
import tarfile
from tqdm import tqdm
from util import resample_volume

base_path = '../../../data/oasis/near/ATLAS'

tar_path = 'ATLAS_R2.0.tar.gz'

ignored_cohorts = ['R027', 'R047', 'R049', 'R050']

if os.path.exists(tar_path):
    tar_output = 'tmp/ATLAS'
    print('Extracting ATLAS data. This may take a while...')
    
    with tarfile.open(tar_path, 'r:gz') as tar_file:
        if not os.path.exists(os.path.join(tar_output, 'ATLAS_2/Training/R052')):
            tar_file.extractall(path=tar_output)

    training_path = os.path.join(tar_output, 'ATLAS_2', 'Training')
    for cohort in tqdm(os.listdir(training_path), desc='Processing ATLAS data'):
        if cohort in ignored_cohorts:
            continue
    
        cohort_path = os.path.join(training_path, cohort)
        cohorts = os.listdir(cohort_path)
        subjects = list(cohorts)

        for idx, subject in tqdm(enumerate(subjects)):
            subject_path = os.path.join(cohort_path, subject)
            session_path = f'{subject_path}/ses-1/anat'

            if 'dataset_description.json' in session_path:
                continue
            for image in os.listdir(session_path):
                if 'T1w' in image and 'resampled' not in image:
                    image_path = os.path.join(session_path, image)
                    path_dir = os.path.dirname(image_path)
                    path_filename = os.path.basename(image_path).split('.')[0]
                    resampled_output = f"{path_dir}/{path_filename}_resampled.nii.gz"
                    if not os.path.exists(resampled_output):
                        # Resample to isotropic 1mm^3
                        resample_volume(image_path, resampled_output, new_spacing=[1.0, 1.0, 1.0])

                    skull_stripped_output = f"{base_path}/{cohort}/{subject}/ses-1/anat/{path_filename}_resampled_skull_stripped.nii.gz"

                    print(skull_stripped_output)
                    os.makedirs(os.path.dirname(skull_stripped_output), exist_ok=True)

                    if not os.path.exists(skull_stripped_output):
                    # Apply skull stripping
                        command = f'hd-bet -i "{resampled_output}" -o "{skull_stripped_output}"'
                        os.system(command)
                    else:
                        pass
                        #print(f"Skull stripped file {skull_stripped_output} already exists")
    
    print('CATARACTS: Processing complete')
else:
    print('To acquire the ATLAS data, please visit https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html and agree to their terms of use.\nAfterwards, follow the download instructions.\nAfter filling out the brief Google form at https://goo.gl/forms/KwCljKSLWbbHWalD2 and acquiring the decryption key, place the decrypted "ATLAS_R2.0.tar.gz" in the directory of this script and run this script again.')
    print('Also, make sure that HD-BET (used for brain extraction) is installed (pip install hd-bet, to reproduce paper results, version 2.0.1 is required) and callable from the command line using "hd-bet".')

