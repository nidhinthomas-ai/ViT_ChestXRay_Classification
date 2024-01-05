# prepare_data.py
import os
import pandas as pd

def main():
    # Paths
    metadata_path = "path_to_dataset/Chest_xray_Corona_Metadata.csv"
    data_dir = "path_to_dataset/data1"
    old_data_dir = "path_to_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"

    # Read metadata
    metadata = pd.read_csv(metadata_path, index_col=0)
    
    # Filter metadata for normal and virus categories
    normal_meta = metadata[metadata["Label"] == "Normal"]
    virus_meta = metadata[metadata["Label_1_Virus_category"] == "Virus"]

    # Directory setup
    normal_dir = f"{data_dir}/normal"
    virus_dir = f"{data_dir}/virus"
    dir_dict = {
        normal_dir: normal_meta,
        virus_dir: virus_meta,
    }
    os.makedirs(data_dir, exist_ok=True)

    # Process and organize images into respective directories
    for dir, meta in dir_dict.items():
        os.makedirs(dir, exist_ok=True)
        for im in meta["X_ray_image_name"]:
            src_train = f"{old_data_dir}/train/{im}"
            src_test = f"{old_data_dir}/test/{im}"
            dst = f"{dir}/{im}"
            if os.path.exists(src_train):
                os.rename(src_train, dst)
            elif os.path.exists(src_test):
                os.rename(src_test, dst)

if __name__ == "__main__":
    main()
