import os
import time
import zipfile
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------------
# 1. Hardware Optimization for Raspberry Pi 400 (Quad-Core)
# ---------------------------------------------------------
torch.set_num_threads(4)  # Utilize all 4 Cortex-A72 cores
DEVICE = torch.device("cpu")

# Must match the alphabetical folder order ImageFolder used during training:
# COVID, Normal, Pneumonia
CLASSES = ["COVID-19", "Normal", "Pneumonia"]

# ---------------------------------------------------------
# 2. Custom BTP CNN Architecture (Fixed Shapes)
# ---------------------------------------------------------
class CustomBTPNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomBTPNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Classifier (Processes 50176 flattened features into 512 nodes)
        self.fc1 = nn.Linear(50176, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 3. Model Initializers (Modified for 3 classes)
# ---------------------------------------------------------
def get_densenet121(num_classes=3):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def get_resnet50(num_classes=3):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_efficientnet_b0(num_classes=3):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_vgg16(num_classes=3):
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

# ---------------------------------------------------------
# 4. Safe Checkpoint Loading & Repacking
# ---------------------------------------------------------
def repack_extracted_checkpoint(extracted_pkl_path):
    """Zips an unzipped PyTorch checkpoint back into a valid .pt archive with a 2026 timestamp."""
    model_dir = os.path.dirname(extracted_pkl_path)
    model_name = os.path.basename(os.path.normpath(model_dir))
    repacked_pt_path = f"{model_name}_repacked.pt"
    
    if os.path.exists(repacked_pt_path):
        return repacked_pt_path

    print(f"  -> Repacking extracted folder '{model_dir}' into '{repacked_pt_path}'...")
    with zipfile.ZipFile(repacked_pt_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith("_repacked.pt"):
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_dir)
                
                zinfo = zipfile.ZipInfo(f"archive/{arcname}")
                zinfo.date_time = (2026, 1, 1, 0, 0, 0)
                zinfo.compress_type = zipfile.ZIP_STORED
                
                with open(file_path, 'rb') as f:
                    zipf.writestr(zinfo, f.read())
                
    return repacked_pt_path

def load_checkpoint(model, pkl_or_pth_path):
    print(f"Loading weights from: {pkl_or_pth_path}")
    
    if pkl_or_pth_path.endswith("data.pkl") and os.path.isdir(os.path.dirname(pkl_or_pth_path)):
        valid_checkpoint = repack_extracted_checkpoint(pkl_or_pth_path)
    else:
        valid_checkpoint = pkl_or_pth_path

    # Disable weights_only to bypass PyTorch 2.6 restrictions on legacy archives
    state_dict = torch.load(valid_checkpoint, map_location=DEVICE, weights_only=False)
    
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing and "
              f"{len(result.unexpected_keys)} unexpected keys - "
              "this checkpoint may not match the architecture.")
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------
# 5. Image Transformation & Preprocessing
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 6. Benchmark and Inference Routine
# ---------------------------------------------------------
def run_benchmark(model, model_name, input_tensor, warmup_runs=3, test_runs=10):
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            
        times = []
        for _ in range(test_runs):
            t0 = time.perf_counter()
            output = model(input_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

        avg_latency = sum(times) / len(times)
        fps = 1000.0 / avg_latency
        
    print(f"[{model_name}]")
    print(f"  Prediction : {CLASSES[pred_idx]} ({confidence:.2f}%)")
    print(f"  Avg Latency: {avg_latency:.2f} ms ({fps:.2f} FPS)\n")
        
    return {
        "Prediction" : f"{CLASSES[pred_idx]} ({confidence:.2f}%)",
        "Avg Latency": f"{avg_latency:.2f} ms ({fps:.2f} FPS)",
    }

if __name__ == "__main__":
    # Synthetic test input (batch_size=1, channels=3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    # Dictionary of your models and checkpoint paths
    model_configs = {
        "Custom BTP CNN": (CustomBTPNet(3), "best_custom_btp_net_3class/data.pkl"),
        "DenseNet-121": (get_densenet121(3), "best_densenet121_3class/data.pkl"),
        "ResNet-50": (get_resnet50(3), "best_resnet50_3class/data.pkl"),
        "EfficientNet-B0": (get_efficientnet_b0(3), "best_efficientnet_b0_3class/data.pkl"),
        "VGG-16": (get_vgg16(3), "best_vgg16_3class/data.pkl"),
    }

    print("========================================")
    print("   BTP Edge Inference Benchmark (Pi 400)")
    print("========================================\n")

    for name, (model, path) in model_configs.items():
        if os.path.exists(path):
            load_checkpoint(model, path)
            run_benchmark(model, name, dummy_input)
        else:
            print(f"Skipping {name}: Checkpoint file '{path}' not found.\n")
