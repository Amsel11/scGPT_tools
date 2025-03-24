import requests
from tqdm import tqdm
import json
import os
import sys
import time


def download_cellxgene_data(url, output_dir, file_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if file_name is not None:
        filename = file_name
    else:
        filename = url.split('/')[-1]
    file_path = os.path.join(output_dir, filename)
    if os.path.exists(file_path):
        print(f"File {filename} already exists in {output_dir}")
        return file_path
    
    try:
        print(f"Downloading {filename} from {url} to {output_dir}")
        response = requests.get(url, stream=True, timeout=30)
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Download completed: {file_path}")
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove partial download
        return None


def main():
    import argparse 
    parser = argparse.ArgumentParser(description='Download CellXGene dataset from URL')
    parser.add_argument('--url', type=str, required=True, help='URL of the cellxgene dataset')
    parser.add_argument('--file_name', type=str, default=None, help="Optional name for the downloaded file")
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    args = parser.parse_args()

    result = download_cellxgene_data(args.url, args.output_dir, args.file_name)
    
    if result:
        print(f"Successfully downloaded to {result}")
        sys.exit(0)
    else:
        print("Download failed")
        sys.exit(1)


if __name__ == '__main__':
    main()