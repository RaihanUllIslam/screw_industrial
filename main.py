from download_data import DataDownloader
from augment_data import DataAugmentor
from train import Trainer
from evaluation import Evaluator
from attention_map import AttentionMapVisualizer
from dataloader import load_split_data
from model import PretrainedModel
import torch
import os
from glob import glob

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    file_id = "11ozVs6zByFjs9viD3VIIP6qKFgjZwv9E"  # Google Drive file ID for dataset
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-5
    grad_clip = None
    freeze_layers = False
    visualize_attention = True
    output_dir = './eval_results'
    test_image_dir = 'dataset/archive/test/'  
    output_attention_dir = './test_predict'  
    train_dir = 'dataset/archive/train'
    good_dir = os.path.join(train_dir, 'good')
    not_good_dir = os.path.join(train_dir, 'not-good')
    output_aug_dir = 'dataset/archive/train-after-aug'
    output_aug_good_dir = os.path.join(output_aug_dir, 'good')
    output_aug_not_good_dir = os.path.join(output_aug_dir, 'not-good-with-aug')

    if not os.path.exists(good_dir) or not os.path.exists(not_good_dir):
        print(f"Dataset not found. Downloading dataset...")
        downloader = DataDownloader(file_id=file_id)
        downloader.download_and_extract()

    if not os.path.exists(good_dir) or not os.path.exists(not_good_dir):
        print(f"Error: Dataset directories '{good_dir}' or '{not_good_dir}' do not exist after download. Exiting...")
        return
    if not os.path.exists(output_aug_good_dir) or not os.path.exists(output_aug_not_good_dir):
        print(f"Performing data augmentation...")
        augmentor = DataAugmentor(not_good_dir, output_aug_dir, 200)  # Augment 200 images
        augmentor.augment_images()
    else:
        print(f"Augmented data already exists in {output_aug_dir}. Skipping augmentation.")

    train_loader, val_loader, test_loader = load_split_data(output_aug_dir, batch_size=batch_size)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    model = PretrainedModel(freeze=freeze_layers).model.to(device)
    trainer = Trainer(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, grad_clip_value=grad_clip, device=device)
    trainer.train()

    evaluator = Evaluator(model, test_loader, output_dir=output_dir, device=device)
    evaluator.evaluate()

    if visualize_attention:
        attention_visualizer = AttentionMapVisualizer(model, model.layer4, device=device)
        if not os.path.exists(output_attention_dir):
            os.makedirs(output_attention_dir)

        test_images = glob(os.path.join(test_image_dir, '*.png'))
        for img_path in test_images:
            img_name = os.path.basename(img_path) 
            output_image_path = os.path.join(output_attention_dir, f'attention_map_{img_name}')
            print(f"Processing {img_name}...")
            attention_visualizer.visualize(input_image_path=img_path, output_image_path=output_image_path)

if __name__ == '__main__':
    main()
