import kagglehub
import kagglehub
import shutil
import os

# Download directory to KaggleHub cache
download_dir = kagglehub.dataset_download("hojjatk/mnist-dataset")

# Copy the whole directory into the current working directory
target_dir = "./mnist-dataset"

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

shutil.copytree(download_dir, target_dir)

print("Dataset directory copied to:", target_dir)
