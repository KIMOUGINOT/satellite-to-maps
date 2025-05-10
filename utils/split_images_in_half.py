import cv2
import os
from tqdm import tqdm

def split(input_dir):
    source_dir = os.path.join(input_dir, "source")
    groundtruth_dir = os.path.join(input_dir, "groundtruth")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc='splitting images...'):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            _, y, _ = image.shape

            source = image[:,:y//2]
            groundtruth = image[:,y//2:]

            cv2.imwrite(os.path.join(source_dir, filename), source)
            cv2.imwrite(os.path.join(groundtruth_dir, filename), groundtruth)

    print("Done ✅")
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split the image in horizontal half.")
    parser.add_argument("input_dir", type=str, help="Input images directory.")
    args = parser.parse_args()
    split(args.input_dir)