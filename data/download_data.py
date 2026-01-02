"""
Script to download and extract MovieLens 100K dataset
"""

import os
import requests
import zipfile
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, DATASET_CONFIG


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def main():
    """Main function to download MovieLens dataset"""
    
    url = DATASET_CONFIG['url']
    zip_filename = 'ml-100k.zip'
    zip_path = os.path.join(RAW_DATA_DIR, zip_filename)
    
    # Check if already downloaded
    extracted_folder = os.path.join(RAW_DATA_DIR, 'ml-100k')
    if os.path.exists(extracted_folder):
        print("Dataset already exists. Skipping download.")
        return extracted_folder
    
    # Download
    print(f"Downloading MovieLens 100K dataset from {url}")
    download_file(url, zip_path)
    
    # Extract
    extract_zip(zip_path, RAW_DATA_DIR)
    
    # Remove zip file
    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to {extracted_folder}")
    
    return extracted_folder


if __name__ == "__main__":
    main()