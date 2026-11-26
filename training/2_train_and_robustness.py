"""
2_train_and_robustness.py
B.Tech Project: Robustness Analysis of CNNs vs ViTs in Medical Imaging
Description: Trains models, optimizes ensemble via Optuna, and evaluates noise robustness.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paper Plot Styling
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})

UNIFIED_PATH = "./btp_3class_dataset"
IMG_SIZE, BATCH = 224, 32

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

loaders = {x: DataLoader(datasets.ImageFolder(f"{UNIFIED_PATH}/{x}", transform), 
                         batch_size=BATCH, shuffle=(x=='train')) for x in ['train', 'val', 'test']}

model_list = ["resnet50", "efficientnet_b0", "densenet121", "vgg16", "vit_base_patch16_224"]

class ClinicalNoise:
    @staticmethod
    def apply_poisson(img, sigma):
        peak = 1.0 / (sigma + 1e-6)
        rate = torch.clamp(img * peak, min=0.0)
        return torch.poisson(rate) / peak

    @staticmethod
    def apply_speckle(img, sigma):
        return img + img * torch.randn_like(img) * sigma

    @staticmethod
    def apply_salt_pepper(img, sigma):
        noisy = img.clone()
        mask = torch.rand_like(img)
        noisy[mask < (sigma/2)] = -2.5
        noisy[mask > (1 - sigma/2)] = 2.5
        return noisy

    @staticmethod
    def apply_gaussian(img, sigma):
        return img + sigma * torch.randn_like(img)

def main():
    print(f"✅ Using Device: {device}")
    
    # 1. Training Phase
    models_dict = {}
    for name in model_list:
        weights_path = f"best_{name}_3class.pth"
        model = timm.create_model(name, pretrained=True, num_classes=3).to(device)
        
        if os.path.exists(weights_path):
            print(f"📥 Loading existing weights for {name}...")
            model.load_state_dict(torch.load(weights_path))
        else:
            print(f"🚀 Training {name}...")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            best_acc = 0.0
            for epoch in range(10):
                model.train()
                for imgs, labels in loaders['train']:
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(imgs), labels)
                    loss.backward()
                    optimizer.step()
                
                # Simple validation checkpointing
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for imgs, labels in loaders['val']:
                        out = model(imgs.to(device))
                        preds.extend(torch.argmax(out, 1).cpu().numpy())
                        targets.extend(labels.numpy())
                acc = accuracy_score(targets, preds)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), weights_path)
            model.load_state_dict(torch.load(weights_path))
        models_dict[name] = model

    # 2. Optuna Ensemble
    val_preds_all, val_targets_list = [], []
    for name in model_list:
        model = models_dict[name]
        model.eval()
        probs = []
        with torch.no_grad():
            for imgs, labels in loaders['val']:
                out = model(imgs.to(device))
                probs.extend(torch.softmax(out, dim=1).cpu().numpy())
                if name == model_list[0]: val_targets_list.extend(labels.numpy())
        val_preds_all.append(np.array(probs))
    
    val_targets = np.array(val_targets_list)
    
    def objective(trial):
        w = np.array([trial.suggest_float(f"w{i}", 0, 1) for i in range(len(model_list))])
        w = w / sum(w)
        ensemble_pred = sum(w[i] * val_preds_all[i] for i in range(len(model_list)))
        return roc_auc_score(val_targets, ensemble_pred, average="macro", multi_class='ovr')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    best_w = np.array([study.best_params[f"w{i}"] for i in range(len(model_list))])
    best_w /= best_w.sum()

    # 3. Robustness Evaluation
    noise_types = ["Poisson", "Speckle", "Salt&Pepper", "Gaussian"]
    levels = [0.01, 0.05, 0.1, 0.2]
    plot_models = model_list + ["Optimized_Ensemble"]
    robustness_results = {name: {nt: [] for nt in noise_types} for name in plot_models}

    for nt in noise_types:
        for lvl in levels:
            test_preds = {name: [] for name in model_list}
            test_targets_list = []
            with torch.no_grad():
                for imgs, labels in loaders['test']:
                    imgs = imgs.to(device)
                    if nt == "Poisson": noisy = ClinicalNoise.apply_poisson(imgs, lvl)
                    elif nt == "Speckle": noisy = ClinicalNoise.apply_speckle(imgs, lvl)
                    elif nt == "Salt&Pepper": noisy = ClinicalNoise.apply_salt_pepper(imgs, lvl)
                    else: noisy = ClinicalNoise.apply_gaussian(imgs, lvl)
                    
                    test_targets_list.extend(labels.numpy())
                    for name in model_list:
                        out = models_dict[name](noisy)
                        test_preds[name].extend(torch.softmax(out, dim=1).cpu().numpy())
            
            for name in model_list:
                preds_class = np.argmax(np.array(test_preds[name]), axis=1)
                robustness_results[name][nt].append(accuracy_score(test_targets_list, preds_class))
                
            ensemble_probs = sum(best_w[i] * np.array(test_preds[name]) for i, name in enumerate(model_list))
            robustness_results["Optimized_Ensemble"][nt].append(accuracy_score(test_targets_list, np.argmax(ensemble_probs, axis=1)))

    # 4. Plotting
    colors = sns.color_palette("colorblind", len(model_list))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, nt in enumerate(noise_types):
        ax = axes[i//2, i%2]
        for idx, m_name in enumerate(model_list):
            ax.plot(levels, robustness_results[m_name][nt], marker='s', linestyle='--', alpha=0.7, color=colors[idx], label=m_name)
        ax.plot(levels, robustness_results["Optimized_Ensemble"][nt], marker='o', linewidth=2.5, color='black', label="Optimized Ensemble")
        ax.set_title(f"Robustness Degradation: {nt} Noise", fontweight='bold')
        ax.set_xlabel(r"Noise Severity ($\sigma$)")
        ax.set_ylabel("Classification Accuracy")
        if i == 0: ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("Fig1_Ensemble_Robustness_Curves.pdf", dpi=300)
    print("✅ Evaluation complete. Saved Fig1_Ensemble_Robustness_Curves.pdf")

if __name__ == "__main__":
    main()
