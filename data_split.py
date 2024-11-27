# Collect all train_and_val cases and split into train_and_val and validation sets

import os
import random
import shutil

# Create current_god_run and val folders
train_images_dir = os.path.join('data', 'current_god_run', 'images')
train_labels_dir = os.path.join('data', 'current_god_run', 'labels')
val_images_dir = os.path.join('data', 'val', 'images')
val_labels_dir = os.path.join('data', 'val', 'labels')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List all image files (assumes images are in folder_path and labels have the same name but .txt extension)
image_files = [f for f in os.listdir('data/train_and_val/images') if f.endswith(('.jpg', '.png'))]

# Shuffle and split data
random.seed(0)
random.shuffle(image_files)
split_idx = int(len(image_files) * (1 - 0.2))  # 20% validation data
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Copy files to respective folders
for file_list, images_dir, labels_dir in [(train_files, train_images_dir, train_labels_dir),
                                          (val_files, val_images_dir, val_labels_dir)]:
    for image_file in file_list:
        image_path = os.path.join('data/train_and_val/images', image_file)
        label_path = os.path.join('data/train_and_val/labels', image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Copy image
        shutil.copy(image_path, os.path.join(images_dir, image_file))
        # Copy corresponding label file (if it exists)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(labels_dir, os.path.basename(label_path)))

print(f"Dataset split complete. Training cases: {len(train_files)}, Validation cases: {len(val_files)}")
