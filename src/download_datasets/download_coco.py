# import the required modules to download the datasets
import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url: str, dest: str) -> None:
    """
    Downloads the file from the given URL to the specified destination
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

def main():
    # create directory if it does not exist
    os.makedirs('datasets/coco', exist_ok=True)
    
    # Define file paths
    train_zip_path = 'datasets/coco/train2017.zip'
    annotations_zip_path = 'datasets/coco/annotations_trainval2017.zip'
    train_extracted_path = 'datasets/coco/train2017'
    annotations_extracted_path = 'datasets/coco/annotations'

    # Download the COCO Dataset if not already downloaded
    if not os.path.exists(train_zip_path):
        download_file('http://images.cocodataset.org/zips/train2017.zip', train_zip_path)
    if not os.path.exists(annotations_zip_path):
        download_file('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', annotations_zip_path)

    # Extract the Dataset if not already extracted
    if not os.path.exists(train_extracted_path):
        with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
            zip_ref.extractall('datasets/coco/')
    if not os.path.exists(annotations_extracted_path):
        with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
            zip_ref.extractall('datasets/coco/')

if __name__ == '__main__':
    main()