import os
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO

class WasteDatasetLoader:
    """Unified loader for waste classification datasets"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.trashnet_dir = Path(r"C:\Users\aondo\OneDrive\Documents\semester three fanshawe college\INFO 6156 Capstone Project\main_capstone\datasets\custom_waste\trashnet\dataset-resized")
        self.coco_dir = Path(r"C:\Users\aondo\OneDrive\Documents\semester three fanshawe college\INFO 6156 Capstone Project\main_capstone\datasets\coco")
        self.categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    
    def load_trashnet(self):
        """Load TrashNet dataset"""
        images = []
        labels = []
        
        print("Loading TrashNet dataset...")
        for label, category in enumerate(self.categories):
            category_path = self.trashnet_dir / category
            if category_path.exists():
                for img_path in category_path.glob('*.jpg'):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, self.image_size)
                            images.append(img)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels)

    def load_coco(self):
        """Load COCO dataset"""
        images = []
        labels = []
        
        print("Loading COCO dataset...")
        ann_file = self.coco_dir / "annotations" / "instances_train2017.json"
        if ann_file.exists():
            coco = COCO(str(ann_file))
            
            # Map COCO categories to our categories
            category_map = {
                'bottle': 0,  # glass
                'book': 1,    # paper
                'box': 2,     # cardboard
                'plastic bag': 3,  # plastic
                'can': 4,     # metal
                'other': 5    # trash
            }
            
            for img_id in coco.getImgIds():
                try:
                    img_info = coco.loadImgs(img_id)[0]
                    img_path = self.coco_dir / "train2017" / img_info['file_name']
                    
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, self.image_size)
                            
                            # Get annotations for this image
                            ann_ids = coco.getAnnIds(imgIds=img_id)
                            anns = coco.loadAnns(ann_ids)
                            
                            if anns:
                                cat_id = anns[0]['category_id']
                                cat_name = coco.loadCats([cat_id])[0]['name']
                                if cat_name in category_map:
                                    images.append(img)
                                    labels.append(category_map[cat_name])
                                    
                except Exception as e:
                    print(f"Error loading COCO image {img_id}: {str(e)}")
        
        return np.array(images), np.array(labels)
    
    def load_combined_datasets(self):
        """Load and combine both datasets"""
        trashnet_images, trashnet_labels = self.load_trashnet()
        
        try:
            coco_images, coco_labels = self.load_coco()
            
            # Debug information
            print(f"TrashNet images shape: {trashnet_images.shape}")
            print(f"COCO images shape: {coco_images.shape if len(coco_images) > 0 else 'empty'}")
            
            if len(coco_images) > 0 and coco_images.ndim == trashnet_images.ndim:
                # Only combine if dimensions match
                combined_images = np.concatenate([trashnet_images, coco_images])
                combined_labels = np.concatenate([trashnet_labels, coco_labels])
            else:
                print("Warning: COCO dataset has incorrect dimensions. Using only TrashNet.")
                combined_images = trashnet_images
                combined_labels = trashnet_labels
        except Exception as e:
            print(f"Error loading COCO dataset: {e}. Using only TrashNet.")
            combined_images = trashnet_images
            combined_labels = trashnet_labels
        
        print(f"Total images loaded: {len(combined_images)}")
        print(f"Images per category: {np.bincount(combined_labels)}")
        
        return combined_images, combined_labels