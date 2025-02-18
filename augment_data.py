import os
import shutil
from glob import glob
from torchvision import transforms
import random
from PIL import Image

class DataAugmentor:
    def __init__(self, input_folder, output_folder, num_augmentations):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_augmentations = num_augmentations

    def augment_images(self):
        good_aug_dir = os.path.join(self.output_folder, 'good')
        not_good_aug_dir = os.path.join(self.output_folder, 'not-good-with-aug')
        if not os.path.exists(good_aug_dir):
            os.makedirs(good_aug_dir)
        if not os.path.exists(not_good_aug_dir):
            os.makedirs(not_good_aug_dir)
        original_good_images = glob(os.path.join('dataset/archive/train/good', '*.png'))
        for img_path in original_good_images:
            shutil.copy(img_path, good_aug_dir)
        print(f"Copied {len(original_good_images)} original good images to {good_aug_dir}")

        original_not_good_images = glob(os.path.join(self.input_folder, '*.png'))
        for img_path in original_not_good_images:
            shutil.copy(img_path, not_good_aug_dir)
        print(f"Copied {len(original_not_good_images)} original not-good images to {not_good_aug_dir}")

        self.perform_augmentation(not_good_aug_dir, self.num_augmentations)

    def perform_augmentation(self, image_dir, num_augmentations):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomVerticalFlip()
        ])

        images = glob(os.path.join(image_dir, '*.png'))
        for img_path in images:
            original_img = Image.open(img_path)
            for i in range(num_augmentations // len(images)):
                augmented_img = transform(original_img)
                new_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{random.randint(1, 10000)}.png"
                augmented_img.save(os.path.join(image_dir, new_img_name))

        print(f"Augmentation complete! {num_augmentations} augmented images saved in: {image_dir}")
