import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

class AttentionMapVisualizer:
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, module, input, output):
        self.activations = output

    def generate_attention_map(self, input_image):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_image)
        if self.activations is None:
            raise ValueError("Activations not captured. Ensure forward pass is complete.")

        activations = self.activations.squeeze(0).cpu().numpy()
        heatmap = np.mean(activations, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        return heatmap

    def overlay_heatmap(self, heatmap, original_image):
        heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_image = np.array(original_image)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        overlayed_image = cv2.addWeighted(heatmap, 0.5, original_image_rgb, 0.5, 0)
    
        return overlayed_image


    def visualize(self, input_image_path, output_image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        original_image = Image.open(input_image_path).convert("L")
        input_image = transform(original_image).unsqueeze(0).to(self.device)
        heatmap = self.generate_attention_map(input_image)
        overlayed_image = self.overlay_heatmap(heatmap, original_image)
        cv2.imwrite(output_image_path, overlayed_image)
        plt.imshow(overlayed_image)
        plt.axis('off')
        plt.show()
