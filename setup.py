import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='OpenMIBOOD',
    version='1.0',
    author='OpenMIBOOD dev team',
    author_email='Github@re-mic.de',
    description=
    'This package (https://github.com/remic-othr/OpenMIBOOD) extends the OpenOOD framework from https://github.com/Jingkang50/OpenOOD with three medical imaging benchmarks for Out-Of-Distribution detection.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.13.1',
        'torchvision>=0.13',
        'scikit-learn',
        'json5',
        'matplotlib',
        'scipy',
        'tqdm',
        'pyyaml>=5.4.1',
        'pre-commit',
        'opencv-python>=4.4.0.46',
        'imgaug>=0.4.0',
        'pandas',
        'diffdist>=0.1',
        'Cython>=0.29.30',
        'faiss-gpu>=1.7.2',
        'gdown>=4.7.1',  # 'libmr>=0.1.9'
        'nnunetv2==2.5.2',
        'hd-bet==2.0.1',
        'nibabel',
        'pydicom',
        'libmr>=0.1.9',
        'torchio>=0.18.0',
        'monai>=0.7.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
)
