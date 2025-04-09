import requests
from tqdm import tqdm
import json
import os
import sys
import time


def download_cellxgene_data(dataset_id, output_dir='data'):
    # Download with progress bar if file doesn't exist
    if not file_path.exists():
        print(f"Downloading {dataset_name} from {dataset_download_url} to {file_path}")
        response = requests.get(dataset_download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
    # Update tracking file


def main():
    import argparse 
    parser = argparse.ArgumentParser()
    
    download_cellxgene_data('c605e7df-96a3-40cd-8eb5-53b32dfa9a10')
        