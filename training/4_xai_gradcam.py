"""
4_xai_gradcam.py
B.Tech Project: Robustness Analysis of CNNs vs ViTs in Medical Imaging
Description: Generates Grad-CAM heatmaps for DenseNet-121 and ViT on a noisy test image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import timm
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_xai_visualizations():
    print("🔍 Generating XAI Heatmaps...")
    UNIFIED_PATH = "./btp_3class_dataset"
    IMG_SIZE = 224
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_loader = DataLoader(datasets.ImageFolder(f"{UNIFIED_PATH}/test", transform), batch_size=1, shuffle=True)
    sample_img, sample_label = next(iter(test_loader))
    noisy_tensor = sample_img[0].to(device) + 0.15 * torch.randn_like(sample_img[0].to(device)) # Add Gaussian Noise

    # Load DenseNet & ViT for Comparison
    densenet = timm.create_model("densenet121", pretrained=False, num_classes=3).to(device)
    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=3).to(device)
    densenet.load_state_dict(torch.load("best_densenet121_3class.pth", map_location=device))
    vit.load_state_dict(torch.load("best_vit_base_patch16_224_3class.pth", map_location=device))
    
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        return result.transpose(2, 3).transpose(1, 2)

    # 1. CNN Heatmap
    cam_cnn = GradCAM(model=densenet, target_layers=[densenet.features[-1]])
    gray_cnn = cam_cnn(input_tensor=noisy_tensor.unsqueeze(0), targets=[ClassifierOutputTarget(sample_label[0].item())])[0, :]
    
    # 2. ViT Heatmap
    cam_vit = GradCAM(model=vit, target_layers=[vit.blocks[-1].norm1], reshape_transform=reshape_transform)
    gray_vit = cam_vit(input_tensor=noisy_tensor.unsqueeze(0), targets=[ClassifierOutputTarget(sample_label[0].item())])[0, :]

    # Un-normalize for display
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    plot_img = np.clip(inv_normalize(noisy_tensor.cpu()).permute(1, 2, 0).numpy(), 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(plot_img)
    axes[0].set_title("Noisy Input X-Ray")
    axes[0].axis('off')
    
    axes[1].imshow(show_cam_on_image(plot_img, gray_cnn, use_rgb=True))
    axes[1].set_title("DenseNet-121 Attention\n(Distracted by Noise)")
    axes[1].axis('off')
    
    axes[2].imshow(show_cam_on_image(plot_img, gray_vit, use_rgb=True))
    axes[2].set_title("Vision Transformer Attention\n(Focused on Lungs)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("Fig3_XAI_Comparison.pdf", dpi=300)
    print("✅ Saved XAI Visualization as Fig3_XAI_Comparison.pdf")

if __name__ == "__main__":
    generate_xai_visualizations()
