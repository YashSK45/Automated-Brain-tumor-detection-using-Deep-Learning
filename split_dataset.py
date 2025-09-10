import os
import shutil
import random

def split_data(source_dir, dest_dir, split_ratio=0.8):
    categories = ['yes', 'no']
    for category in categories:
        files = os.listdir(os.path.join(source_dir, category))
        random.shuffle(files)

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        valid_files = files[split_idx:]

        for split_type, split_files in zip(['train', 'valid'], [train_files, valid_files]):
            split_path = os.path.join(dest_dir, split_type, category)
            os.makedirs(split_path, exist_ok=True)
            for file in split_files:
                shutil.copy(os.path.join(source_dir, category, file), os.path.join(split_path, file))

source = 'data'
destination = 'classification_dataset'
split_data(source, destination)
