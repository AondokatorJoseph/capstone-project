import os
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
from huggingface_hub import hf_hub_download

class TrashNetDownloader:
    """Class to handle TrashNet dataset download from Hugging Face"""
    
    def __init__(self):
        self.data_dir = Path("data/custom_waste/trashnet")
        self.repo_id = "garythung/trashnet"
        self.categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.data_dir,
            self.data_dir / "dataset",
            *[self.data_dir / "dataset" / cat for cat in self.categories]
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def download_dataset(self):
        """Download and extract TrashNet dataset from Hugging Face"""
        print("Setting up TrashNet directories...")
        self.setup_directories()
        
        try:
            print("\nDownloading TrashNet dataset from Hugging Face...")
            # Download the dataset archive
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="dataset-resized.zip",
                repo_type="dataset"
            )
            
            print("\nExtracting dataset...")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print("Successfully downloaded and extracted TrashNet dataset!")
            print(f"Dataset location: {self.data_dir.absolute()}")
            
        except Exception as e:
            print(f"Error downloading TrashNet dataset: {str(e)}")

def main():
    # Install required package if not present
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub"])
    
    downloader = TrashNetDownloader()
    downloader.download_dataset()

if __name__ == "__main__":
    main()