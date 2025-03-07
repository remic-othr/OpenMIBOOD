import os
import zipfile
from util import download_with_curl

benchmarks = ['midog', 'phakir', 'oasis3']

for benchmark in benchmarks:
    print(f'Downloading {benchmark} classifier...')
    url = f'https://zenodo.org/records/14982267/files/{benchmark}_classifier.pth?download=1'
    download_with_curl(url, f'tmp/{benchmark}_classifier.pth')
    if not os.path.exists(f'../../../results/{benchmark}/{benchmark}_classifier.pth'):
        os.makedirs(f'../../../results/{benchmark}', exist_ok=True)
        os.rename(f'tmp/{benchmark}_classifier.pth', f'../../../results/{benchmark}/{benchmark}_classifier.pth')
        