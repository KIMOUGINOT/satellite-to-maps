from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import cv2

class SatelliteDataset(Dataset):
    def __init__(self, dataset_path: str, source_dir: str = 'source', groundtruth_dir: str = 'groundtruth', image_size = (256,256)):
        self.dataset_path = dataset_path
        self.source_path = os.path.join(self.dataset_path, source_dir) 
        self.groundtruth_path = os.path.join(self.dataset_path, groundtruth_dir) 
        self.image_size = image_size
        self.db = self._build_db()

    def _build_db(self) -> list:
        print(f"[INFO] Building dataset from {self.dataset_path} ...")
        db = []
        try:
            for filename in tqdm(os.listdir(self.source_path)):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    item = {
                        'source': os.path.join(self.source_path, filename),
                        'groundtruth': os.path.join(self.groundtruth_path, filename)
                    }
                db.append(item)
            return db
        except:
            print(f"[WARNING] No dataset has been found in {self.dataset_path}")
            return db


    def __getitem__(self, index):
        sample = self.db[index]

        source = cv2.imread(sample['source'])
        if source is None:
            raise FileNotFoundError(f"Image not found: {sample['source']}")
        source = cv2.resize(source, self.image_size, interpolation=cv2.INTER_LINEAR)

        groundtruth = cv2.imread(sample['groundtruth'])
        if groundtruth is None:
            raise FileNotFoundError(f"Image not found: {sample['groundtruth']}")
        groundtruth = cv2.resize(groundtruth, self.image_size, interpolation=cv2.INTER_LINEAR)

        return source, groundtruth
    
    def __len__(self) -> int:
        return len(self.db)
    
# if __name__ == "__main__":
#     dataset = SatelliteDataset("datasets/maps/val")
#     src, gt = dataset[0]
#     print(f"Source image shape: {src.shape}")
#     print(f"Groundtruth image shape: {gt.shape}")
#     print(f"Dataset length: {len(dataset)}")