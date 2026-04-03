import rasterio
import numpy as np
import os
from rasterio.enums import Resampling
from glob import glob
import shutil


# --- CONFIGURATION CLASS ---
class PreprocessingConfig:
    """Configuration parameters for the glacier data preprocessing pipeline."""
    
    # OUTPUT PATHS
    OUTPUT_BASE_DIR = 'data_switzerland/preprocessed_data' 
    OUTPUT_X_DIR = os.path.join(OUTPUT_BASE_DIR, 'inputs_X')
    OUTPUT_Y_DIR = os.path.join(OUTPUT_BASE_DIR, 'masks_Y')

    # SPECTRAL INDICES PARAMETERS
    NDSI_THRESHOLD = 0.25      # Threshold to define ice/snow
    NDWI_WATER_THRESHOLD = 0.2 # Filter out water bodies (NDWI < Threshold)
    TILE_SIZE = 512            # Output tile dimension
    
    # FILTER PARAMETERS
    EMPTY_TILE_THRESHOLD = 0.95  # Discard tiles with >95% no-data (black borders)
    MIN_ICE_PERCENTAGE = 0.001    # Discard tiles with <0.1% ice or 100% ice (Boundary Filter)

# --- HELPER FUNCTIONS (STAP 1-3) ---

def stack_and_resample_bands(granule_path, bands_to_stack=['B04', 'B03', 'B02', 'B08', 'B11', 'B12']):
    """Stacks 6 bands (10m and resampled 20m) into a single NumPy array."""
    
    img_data_path = os.path.join(granule_path, 'IMG_DATA')
    r10m_path = os.path.join(img_data_path, 'R10m')
    stacked_bands = []

    try:
        # Determine reference resolution (B04)
        ref_file_matches = [f for f in os.listdir(r10m_path) if 'B04_' in f and f.endswith('.jp2')]
        if not ref_file_matches: return None
        ref_filepath = os.path.join(r10m_path, ref_file_matches[0])
        with rasterio.open(ref_filepath) as ref_src: target_shape = ref_src.shape

        # Loop through bands
        for band_id in bands_to_stack:
            current_path = r10m_path if band_id in ['B04', 'B03', 'B02', 'B08'] else os.path.join(img_data_path, 'R20m')
            filename_matches = [f for f in os.listdir(current_path) if f'{band_id}_' in f and f.endswith('.jp2')]
            if not filename_matches: return None
            filepath = os.path.join(current_path, filename_matches[0])

            with rasterio.open(filepath) as src:
                band_data = src.read(1)
                # Resample 20m bands (B11, B12)
                if band_id in ['B11', 'B12']: 
                    band_data_resampled = src.read(out_shape=(1, target_shape[0], target_shape[1]), resampling=Resampling.bilinear)[0]
                    stacked_bands.append(band_data_resampled)
                else:
                    stacked_bands.append(band_data)

        if stacked_bands: 
            stacked_array = np.stack(stacked_bands)
            return np.transpose(stacked_array, (1, 2, 0)) # Return as (H, W, C)
        return None
    except Exception as e:
        print(f"Error processing {granule_path}: {e}")
        return None

def create_filtered_ndsi_mask(stacked_array_X, config):
    """Creates the NDSI/NDWI label mask."""
    
    GREEN_BAND_INDEX = 1; NIR_BAND_INDEX = 3; SWIR1_BAND_INDEX = 4; EPSILON = 1e-8
    
    B03 = stacked_array_X[:, :, GREEN_BAND_INDEX].astype(np.float32)
    B08 = stacked_array_X[:, :, NIR_BAND_INDEX].astype(np.float32)
    B11 = stacked_array_X[:, :, SWIR1_BAND_INDEX].astype(np.float32)
    
    # Normalization (Memory-safe chaining)
    np.divide(B03, 10000.0, out=B03); np.divide(B08, 10000.0, out=B08); np.divide(B11, 10000.0, out=B11)

    # Calculate NDSI
    NDSI_num = np.subtract(B03, B11); NDSI_den = np.add(B03, B11)
    np.divide(NDSI_num, np.add(NDSI_den, EPSILON), out=NDSI_num)
    NDSI = NDSI_num 

    # Calculate NDWI (Water Filter)
    NDWI_num = np.subtract(B03, B08); NDWI_den = np.add(B03, B08)
    np.divide(NDWI_num, np.add(NDWI_den, EPSILON), out=NDWI_num)
    NDWI = NDWI_num

    # Combine Filters: (NDSI > Threshold) AND (NDWI < Threshold)
    is_ice_candidate = (NDSI > config.NDSI_THRESHOLD)
    is_not_water = (NDWI < config.NDWI_WATER_THRESHOLD) 
    
    label_mask = np.where(is_ice_candidate & is_not_water, 1, 0).astype(np.uint8)
    return np.expand_dims(label_mask, axis=-1)

def tile_and_save_filtered(image_array_X, image_array_Y, scene_id, config):
    """Splits the arrays and filters out empty, pure background and pure ice tiles."""
    
    height, width, channels = image_array_X.shape
    tile_count = 0; discarded_empty = 0; boundary_discarded = 0

    # normalise tiles
    image_array_X_normalized = image_array_X.astype(np.float32) / 10000.0
    total_tile_pixels = config.TILE_SIZE * config.TILE_SIZE
    min_ice_pixels_threshold = total_tile_pixels * config.MIN_ICE_PERCENTAGE 

    for y in range(0, height, config.TILE_SIZE):
        for x in range(0, width, config.TILE_SIZE):
            tile_X = image_array_X_normalized[y:y + config.TILE_SIZE, x:x + config.TILE_SIZE, :]
            tile_Y = image_array_Y[y:y + config.TILE_SIZE, x:x + config.TILE_SIZE, :]

            if tile_X.shape[0] == config.TILE_SIZE and tile_X.shape[1] == config.TILE_SIZE:
                
                # FILTER 1: No-Data Filter (Empty/Black)
                zero_pixels = np.count_nonzero(tile_X == 0)
                if (zero_pixels / (total_tile_pixels * channels)) > config.EMPTY_TILE_THRESHOLD:
                    discarded_empty += 1
                    continue 

                # FILTER 2/3: Boundary Filter (1% Rule)
                ice_pixels = np.sum(tile_Y)
                is_below_threshold = (ice_pixels < min_ice_pixels_threshold)
                is_pure_ice = (ice_pixels == total_tile_pixels)

                if is_below_threshold or is_pure_ice:
                    boundary_discarded += 1
                    continue
                
                # Save the Selected Tile
                tile_id = f"{scene_id}_{tile_count:04d}"
                np.save(os.path.join(config.OUTPUT_X_DIR, f"{tile_id}_X.npy"), tile_X)
                np.save(os.path.join(config.OUTPUT_Y_DIR, f"{tile_id}_Y.npy"), tile_Y)
                
                tile_count += 1
                
    return tile_count, discarded_empty, boundary_discarded

# --- MAIN PREPROCESSING FUNCTION ---

def preprocess_gletscher_data(data_base_dir, config):
    """
    Main function to preprocess all Sentinel-2 scenes in a base directory.

    Args:
        data_base_dir (str): Path to the directory containing all .SAFE folders.
        config (PreprocessingConfig): Configuration object with all parameters.
    """
    
    # 1. Clean up and create output directories
    if os.path.exists(config.OUTPUT_BASE_DIR):
        shutil.rmtree(config.OUTPUT_BASE_DIR)
        print(f"Removed previous output directory: {config.OUTPUT_BASE_DIR}")
        
    os.makedirs(config.OUTPUT_X_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_Y_DIR, exist_ok=True)
    
    # Find all Sentinel-2 .SAFE folders
    safe_folders = glob(os.path.join(data_base_dir, '**', '*.SAFE'), recursive=True)
    total_scenes = len(safe_folders)
    
    if total_scenes == 0:
        print(f"FATAL: No .SAFE folders found in {data_base_dir}")
        return

    print(f"Found {total_scenes} scenes. Starting preprocessing...")
    
    # 2. Process each scene
    for i, safe_path in enumerate(safe_folders):
        print(f"\n--- Processing Scene {i+1}/{total_scenes}: {os.path.basename(safe_path)} ---")
        
        # Find the actual GRANULE/DATUM_FOLDER
        granule_dirs = glob(os.path.join(safe_path, 'GRANULE', '*'))
        if not granule_dirs:
            print("Skipping: Could not find GRANULE sub-directory.")
            continue
            
        granule_path = granule_dirs[0] # Use the first (and usually only) date folder

        # STAP 1: Stacking
        stacked_input_X = stack_and_resample_bands(granule_path)
        
        if stacked_input_X is None:
            print("Skipping: Stacking failed for this scene.")
            continue
            
        # STAP 2: Masking
        label_mask_Y = create_filtered_ndsi_mask(stacked_input_X, config)
        
        # STAP 3: Tiling
        scene_identifier = os.path.basename(granule_path).split('_')[3]
        total_tiles, empty_discarded, boundary_discarded = tile_and_save_filtered(
            stacked_input_X, label_mask_Y, scene_identifier, config
        )
        
        print(f"SUMMARY: Tiles saved: {total_tiles} | Empty discarded: {empty_discarded} | Pure discarded: {boundary_discarded}")

    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"All final tiles are stored in the '{config.OUTPUT_BASE_DIR}' folder.")