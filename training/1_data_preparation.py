"""
1_data_preparation.py
B.Tech Project: Robustness Analysis of CNNs vs ViTs in Medical Imaging
Description: Downloads and organizes perfectly balanced 3-class X-ray dataset.
"""

import os
import shutil
import glob
from sklearn.model_selection import train_test_split

def prepare_dataset():
    print("🧹 1. Cleaning up old directories...")
    os.system("rm -rf ./btp_raw_data")
    os.system("rm -rf ./btp_3class_dataset")

    BASE_RAW = "./btp_raw_data"
    UNIFIED_PATH = "./btp_3class_dataset"
    os.makedirs(BASE_RAW, exist_ok=True)

    # Kaggle datasets
    datasets_to_get = {
        "pneumonia": "paultimothymooney/chest-xray-pneumonia",
        "covid": "tawsifurrahman/covid19-radiography-database"
    }

    print("📊 2. Downloading reliable datasets via Kaggle API...")
    for folder, slug in datasets_to_get.items():
        dest = f"{BASE_RAW}/{folder}"
        if not os.path.exists(dest):
            print(f"📥 Downloading {folder} dataset...")
            os.system(f"kaggle datasets download -d {slug} -p {dest}")
            os.system(f"unzip -q {dest}/*.zip -d {dest}")
            os.system(f"rm {dest}/*.zip")
            print(f"✅ {folder} extracted.")

    print("\n🚀 3. Building balanced 3-Class dataset...")
    for split in ["train", "val", "test"]:
        for cls in ["Normal", "COVID", "Pneumonia"]:
            os.makedirs(os.path.join(UNIFIED_PATH, split, cls), exist_ok=True)

    pneumonia_raw_dir = f"{BASE_RAW}/pneumonia/chest_xray"
    sample_size = 1500
    
    normal_imgs = glob.glob(f"{pneumonia_raw_dir}/**/NORMAL/*.jpeg", recursive=True)[:sample_size]
    pneumonia_imgs = glob.glob(f"{pneumonia_raw_dir}/**/PNEUMONIA/*.jpeg", recursive=True)[:sample_size]
    covid_imgs = glob.glob(f"{BASE_RAW}/covid/**/COVID/images/*.png", recursive=True)[:sample_size]

    def split_and_copy(files, class_name):
        if not files: return
        train, temp = train_test_split(files, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        for subset, s_name in zip([train, val, test], ["train", "val", "test"]):
            for f in subset:
                shutil.copy(f, os.path.join(UNIFIED_PATH, s_name, class_name))

    print("Copying images to train/val/test folders...")
    split_and_copy(normal_imgs, 'Normal')
    split_and_copy(pneumonia_imgs, 'Pneumonia')
    split_and_copy(covid_imgs, 'COVID')
    
    print(f"\n🎉 SUCCESS! Dataset organized at: {os.path.abspath(UNIFIED_PATH)}")

if __name__ == "__main__":
    prepare_dataset()
