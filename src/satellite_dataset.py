from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import cv2

class SatelliteDataset(Dataset):
    def __init__(self, dataset_path: str, source_dir: str = 'source', groundtruth_dir: str = 'groundtruth', image_size = (256,256)):
        self.dataset_path = dataset_path
        self.source_path = os.path.join(self.dataset_path, source_dir) 
        self.groundtruth_path = os.path.join(self.dataset_path, groundtruth_dir) 
        self.image_size = (image_size[0], image_size[1])
        self.db = self._build_db()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor(),                      # uint8→float32 [0,1]
            transforms.Normalize((0.5,0.5,0.5),         # → [–1,1]
                                 (0.5,0.5,0.5)),
        ])

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

        source = Image.open(sample['source']).convert('RGB')
        source = self.transform(source)

        target = Image.open(sample['groundtruth']).convert('RGB')
        target = self.transform(target)

        return source, target
    
    def __len__(self) -> int:
        return len(self.db)