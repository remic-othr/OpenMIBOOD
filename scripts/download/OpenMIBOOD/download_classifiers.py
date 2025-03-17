import os
import zipfile
from util import download_with_curl

benchmarks = ['midog', 'phakir', 'oasis3']

script_dir = os.path.dirname(os.path.abspath(__file__))

for benchmark in benchmarks:
    print(f'Downloading {benchmark} classifier...')
    url = f'https://zenodo.org/records/14982267/files/{benchmark}_classifier.pth?download=1'
    download_with_curl(url, f'{script_dir}/tmp/{benchmark}_classifier.pth')
    benchmark_path = os.path.join(script_dir, f'../../../results/{benchmark}')
    classifier_path = os.path.join(benchmark_path, f'{benchmark}_classifier.pth')
    if not os.path.exists(benchmark_path):
        os.makedirs(benchmark_path, exist_ok=True)
        os.rename(f'{script_dir}/tmp/{benchmark}_classifier.pth', classifier_path)
        