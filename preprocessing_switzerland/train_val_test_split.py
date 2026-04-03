import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split 
import random
import numpy as np

def create_nested_dirs(split_dirs_base):
    """Creates the base split directories and the nested /X and /Y folders."""
    for split_base_dir in split_dirs_base.values():
        os.makedirs(os.path.join(split_base_dir, 'X'), exist_ok=True)
        os.makedirs(os.path.join(split_base_dir, 'Y'), exist_ok=True)

def copy_files_to_nested_structure(file_list, source_x_folder, source_y_folder, target_base_dir):
    """
    Copies X files to target_base_dir/X and Y files to target_base_dir/Y
    by robustly replacing the source folder names.
    """
    for x_path in file_list:
        # 1. Determine the correct Y path by replacing the input folder name and file suffix.
        # This fixes the FileNotFoundError.
        y_path = x_path.replace(source_x_folder, source_y_folder).replace('_X.npy', '_Y.npy')
        
        # Check for existence of the critical label file
        if not os.path.exists(y_path):
            # This is the expected location for the Y file. If it's still missing,
            # the original preprocessing step failed to save the file.
            raise FileNotFoundError(f"Label file missing. Expected to find it at: {y_path}")
            
        # Copy Input X file to target_base_dir/X/
        shutil.copy(x_path, os.path.join(target_base_dir, 'X', os.path.basename(x_path)))
        # Copy Label Y file to target_base_dir/Y/
        shutil.copy(y_path, os.path.join(target_base_dir, 'Y', os.path.basename(y_path)))

def setup_and_split_data_nested(input_x_dir, input_y_dir, split_dirs_base, test_val_ratio, test_ratio, seed):
    """
    Executes the data splitting and organization.
    """
    # 1. Directory setup
    create_nested_dirs(split_dirs_base)
        
    all_x_files = sorted(glob(os.path.join(input_x_dir, '*_X.npy')))

    if len(all_x_files) == 0:
        print("FATAL: No .npy files found in the input directory. Check input_x_dir definition.")
        return

    # 2. Split logic (80/10/10)
    train_files, temp_files = train_test_split(all_x_files, test_size=test_val_ratio, random_state=seed)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio, random_state=seed)

    print(f"Total tiles found: {len(all_x_files)}")
    print(f"Training set size: {len(train_files)} ({len(train_files)/len(all_x_files)*100:.1f}%)")
    print(f"Validation set size: {len(val_files)} ({len(val_files)/len(all_x_files)*100:.1f}%)")
    print(f"Test set size: {len(test_files)} ({len(test_files)/len(all_x_files)*100:.1f}%)")

    # 3. Determine source folder names from the passed arguments
    source_x_folder = os.path.basename(input_x_dir) # Extracts 'inputs_X'
    source_y_folder = os.path.basename(input_y_dir) # Extracts 'masks_Y'

    # 4. Execute copying to final nested directories
    copy_files_to_nested_structure(train_files, source_x_folder, source_y_folder, split_dirs_base['train'])
    copy_files_to_nested_structure(val_files, source_x_folder, source_y_folder, split_dirs_base['val'])
    copy_files_to_nested_structure(test_files, source_x_folder, source_y_folder, split_dirs_base['test'])

    print("\nSUCCESS: All data has been split and organized into the nested Train/Validation/Test structure.")