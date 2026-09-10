import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial import Voronoi
import imageio.v2 as imageio
from matplotlib import colormaps
from pathlib import Path
from scipy.spatial.distance import cdist
from functools import lru_cache
import warnings
from scipy.spatial import cKDTree  # For nearest neighbor smoothing
# OpenPIV imports removed - using PIVlab data instead
import shutil
import datetime
from skimage.draw import polygon2mask
from matplotlib.ticker import FormatStrFormatter
import scipy.io  # For reading MATLAB .mat files
from matplotlib.patches import Polygon

# File parameters (default: data next to this script; override for your setup)
_script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(_script_dir, "raw_data", "23C")
piv_storage = os.path.join(_script_dir, "raw_data", "PIV", "PIVlab_23C.mat")  # MATLAB PIVlab results

# --- Trial Mode Parameters ---
trial_t = 1         # The timepoint to use for the trial
run_trial = 0       # Set to 1 to run the trial, then set to 0 to run the full pipeline
trial_shift = 0     # Initial shift value for the trsial (can be adjusted)

# ------------------------------------
# Input Parameters (User-Defined)
# ------------------------------------

# Time parameters
start_t = 1
end_t = 150
 
# Analysis parameters
n_value = 6          # n-fold value for the order parameter
shift_down_value = 0  # Amount to shift centroids down in y-direction (pixels)
# Note: shift_down_value can be positive or negative:
#   - Positive: shifts image DOWN (centroids appear lower in visualization)
#   - Negative: shifts image UP (centroids appear higher in visualization)
#   - Used for periodic boundary conditions and coordinate system alignment

# Fourier analysis parameters
fourier_rect = (700, 900, 800, 1000)  # rectangle for Fourier intensity: (x1, x2, y1, y2)
top_peaks_count = 10                  # Top peaks parameter for Fourier analysis

# Video and histogram parameters
fps = 2                              # Frames per second for video output
separation_of_histogram = 25         # Bin size for y-axis histogram of psi values

# ROI analysis parameters
save_roi_psi_average = True  # Whether to save ROI average ψₙ values

# Analysis mode control
piv_only_mode = False  # If True, skip everything except PIV analysis

# PIV interpolation method
piv_interpolation_method = 'bilinear'  # Options: 'bilinear', 'bicubic', 'inverse_distance', 'nearest'
# - 'bilinear': Linear interpolation using scipy's griddata (recommended)
# - 'bicubic': Cubic interpolation using scipy's griddata (smoother but slower)
# - 'inverse_distance': Weighted average based on distance to all grid points (robust for irregular grids)
# - 'nearest': Fastest, uses only the nearest grid point (original method)

# PIV analysis parameters
# Note: PIVlab velocities are in pixels per frame - no scaling needed

# Boundary condition handling
duplicate_for_boundary = True  # If True, duplicate H5 input data 3x vertically for periodic boundary conditions
# This approach duplicates the raw segmentation data before processing, ensuring that:
#   - Top and bottom edges are naturally connected
#   - ψₙ calculations use continuous data across boundaries
#   - ROI evolution can wrap around image edges
#   - Output images remain original size despite internal 3x processing

# ROI for PIV analysis (user-adjustable)
# roi_polygon: list of (x, y) coordinates (AFTER shifting)
# NOTE: These coordinates are in SHIFTED coordinate system and will be converted to ORIGINAL coordinates for PIV evolution
roi_polygon = [
    (800, 800),
    (800, 1000),
    (1000, 1000),
    (1000, 800)
]  # Example: square, but can be any polygon

# Output control flags - Images and Videos
psi_value_picture = 1                # Save ψₙ colored images? (1: yes, 0: no)
fourier_intensity_picture = 1        # Save Fourier intensity images? (1: yes, 0: no)
psi_value_video = 0                  # Create video from ψₙ images? (1: yes, 0: no)
fourier_intensity_picture_video = 0  # Create video from Fourier images? (1: yes, 0: no)
create_psi_histogram = 1             # Create histogram of psi values along y-axis? (1: yes, 0: no)
create_reciprocal_lattice_overlay = 1 # Generate reciprocal lattice overlay? (1: yes, 0: no)
create_smoothed_psi_value_picture = 1 # Save smoothed ψₙ colored images? (1: yes, 0: no)
create_smoothed_psi_histogram = 1     # Create histogram of smoothed psi values along y-axis? (1: yes, 0: no)
save_roi_fourier_intensity = 1  # 1: save ROI-based Fourier intensity maps, 0: do not
manual_roi_fourier_max = True    # True: use manual maximum for ROI Fourier intensity, False: use dynamic maximum
roi_fourier_max_value = 1e7      # Manual maximum value for ROI Fourier intensity (when manual_roi_fourier_max = True)
roi_fourier_crop_size = 100      # Size of the cropped middle region for ROI Fourier intensity (100x100 pixels)

# Output control flags - Data Arrays and Files
psi_value_array = 1                  # Save the ψₙ image array data? (1: yes, 0: no)
fourier_intensity_array = 1          # Save the Fourier intensity array data? (1: yes, 0: no)
save_fourier_peaks_data = 1          # Save Fourier peaks data (offsets, intensities, distances)? (1: yes, 0: no)
save_fourier_metrics_csv = 1         # Save Fourier metrics table as CSV? (1: yes, 0: no)
save_psi_value_mean_csv = 1          # Save average psi_value per timepoint as CSV? (1: yes, 0: no)

# Performance optimization flags
USE_VECTORIZATION = True             # Use vectorized operations where possible
CACHE_VORONOI = True                 # Cache Voronoi computations for repeated patterns
OPTIMIZE_MEMORY = True               # Use memory-efficient operations

piv_analysis = 0 # NEW: Enable PIV analysis? (1: yes, 0: no)
# Use shift_down_value in the piv subdirectory name for clarity
if piv_analysis:
    piv_output_dir = os.path.join(folder_path, 'piv')
    os.makedirs(piv_output_dir, exist_ok=True)

def load_pivlab_data(mat_file_path):
    """
    Load PIV data from MATLAB PIVlab.mat file.
    Returns: u_velocities, v_velocities, x_coords, y_coords
    """
    try:
        # Check if file exists
        if not os.path.exists(mat_file_path):
            print(f"PIVlab data file not found: {mat_file_path}")
            return None, None, None, None
            
        data = scipy.io.loadmat(mat_file_path)
        
        # Validate required data keys
        if 'u_original' not in data or 'v_original' not in data:
            print(f"PIVlab data file missing required keys: 'u_original' or 'v_original'")
            print(f"Available keys: {list(data.keys())}")
            return None, None, None, None
        
        # Extract velocity fields
        u_velocities = []
        v_velocities = []
        
        # PIVlab stores data as nested arrays
        for i in range(len(data['u_original'])):
            u_velocities.append(data['u_original'][i][0])
            v_velocities.append(data['v_original'][i][0])
        
        # Extract coordinate grids (assuming they're the same for all timepoints)
        x_coords = data['x'][0][0] if 'x' in data else None
        y_coords = data['y'][0][0] if 'y' in data else None
        
        print(f"Loaded PIV data: {len(u_velocities)} timepoints")
        print(f"Velocity field shape: {u_velocities[0].shape if u_velocities else 'No data'}")
        if u_velocities:
            print(f"PIV velocity ranges: u=[{np.min(u_velocities[0]):.4f}, {np.max(u_velocities[0]):.4f}], v=[{np.min(v_velocities[0]):.4f}, {np.max(v_velocities[0]):.4f}] pixels/frame")
            print(f"Note: These are direct pixel displacements per frame")
        
        return u_velocities, v_velocities, x_coords, y_coords
        
    except Exception as e:
        print(f"Error loading PIVlab data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Function removed - no longer needed with direct PIVlab integration

if run_trial == 1:
    psi_prefix = f"Psi_T{trial_t}-{trial_t}_Shift_{trial_shift}"
    def process_segmentation_timepoint(t, folder_path, image_dir, n_value,
                                       save_psi_array=False, list_dir='', psi_prefix="", shift_down_value=0,
                                       duplicate_for_boundary=False):
        # Minimal version for trial
        seg_dir = Path(folder_path)
        pattern = f"*T{t:04d}_Simple Segmentation.h5"
        matches = sorted(seg_dir.glob(pattern))
        if not matches:
            print(f"No segmentation file for T{t:04d} in {seg_dir}")
            return None, None, None, None, None, None
        full_path = matches[0]
        with h5py.File(full_path, "r") as f:
            dataset_name = list(f.keys())[0]
            data = np.array(f[dataset_name])
        data = np.squeeze(data)
        
        # Duplicate H5 data for periodic boundary conditions if enabled
        original_height, original_width = data.shape
        if duplicate_for_boundary:
            data = np.vstack([data, data, data])  # Stack 3 copies of raw H5 data
            print(f"Trial: Duplicated H5 segmentation data for boundary conditions: {(original_height, original_width)} -> {data.shape}")
        else:
            print(f"Trial: Using original H5 segmentation data: {data.shape}")
        
        data_binary = np.where(data == 2, 0, data)
        height_img, width_img = data.shape
        labeled_data = measure.label(data_binary)
        regions = measure.regionprops(labeled_data)
        image_shape = (data.shape[0], data.shape[1], 3)
        img_out = np.full(image_shape, 0.5, dtype=np.float32)
        if len(regions) == 0:
            centroids = np.empty((0, 2))
        else:
            centroids = np.array([r.centroid for r in regions])
        def compute_order_parameter(points, n):
            if len(points) == 0:
                return np.zeros(0, dtype=complex)
            if len(points) == 1:
                return np.array([0j])
            valid_dims = [i for i in range(points.shape[1]) if np.any(points[:, i] != points[0, i])]
            if not valid_dims:
                return np.zeros(points.shape[0], dtype=complex)
            reduced_points = points[:, valid_dims]
            if reduced_points.shape[1] < 2:
                return np.zeros(points.shape[0], dtype=complex)
            vor = Voronoi(reduced_points, qhull_options="QJ")
            N = len(points)
            contributions = np.zeros(N, dtype=complex)
            weights = np.zeros(N, dtype=float)
            for (i, j), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
                if -1 in ridge_vertices:
                    continue
                v0, v1 = vor.vertices[ridge_vertices[0]], vor.vertices[ridge_vertices[1]]
                ridge_length = np.linalg.norm(v0 - v1)
                weight = ridge_length**2
                vec_ij = reduced_points[j] - reduced_points[i]
                theta_ij = np.arctan2(vec_ij[1], vec_ij[0])
                contributions[i] += weight * np.exp(1j * n * theta_ij)
                weights[i] += weight
                theta_ji = np.arctan2(-vec_ij[1], -vec_ij[0])
                contributions[j] += weight * np.exp(1j * n * theta_ji)
                weights[j] += weight
            psi_n = np.where(weights > 0, contributions / weights, 0)
            return psi_n
        psi_n = compute_order_parameter(centroids, n_value)
        psi_n_magnitude = np.abs(psi_n)
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = colormaps["viridis"]
        if centroids.shape[0] > 0:
            colors = cmap(norm(psi_n_magnitude))[:, :3]
            for i, region in enumerate(regions):
                color = colors[i]
                coords = region.coords
                img_out[coords[:, 0], coords[:, 1], :] = color
            # Apply shift to image and centroids for visualization
            if trial_shift != 0:
                img_out = np.roll(img_out, trial_shift, axis=0)
                image_height = data.shape[0]
                centroids[:, 0] = (centroids[:, 0] - trial_shift) % image_height
        
        # Crop image back to original size if boundary duplication was used
        if duplicate_for_boundary:
            # Extract the middle section (original image) from the 3x duplicated image
            start_row = original_height
            end_row = 2 * original_height
            img_out_cropped = img_out[start_row:end_row, :, :]
            print(f"Trial: Cropped image from {img_out.shape} to original size {img_out_cropped.shape}")
        else:
            img_out_cropped = img_out
            
        return data_binary, None, img_out_cropped, centroids, psi_n, regions
    t = trial_t
    _, _, img_out, centroids, psi_n, regions = process_segmentation_timepoint(
        t, folder_path, None, n_value,
        save_psi_array=False, list_dir='', psi_prefix=psi_prefix, shift_down_value=trial_shift,
        duplicate_for_boundary=duplicate_for_boundary
    )
    if img_out is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_out, extent=[0, img_out.shape[1], 0, img_out.shape[0]])
        ax.set_title(f"Trial Timepoint {t:04d} - ψ_{n_value} map (Shifted by {trial_shift} pixels)")
        ax.set_xlabel('X-coordinate (pixels)')
        ax.set_ylabel('Y-coordinate (pixels)')
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.set_xticks(np.arange(0, img_out.shape[1]+1, 200))
        ax.set_yticks(np.arange(0, img_out.shape[0]+1, 200))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5, which='both')
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=colormaps["viridis"])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label(r"$|\psi_%d|$" % n_value)
        plt.show()
        plt.close(fig)
    print("Inspect the plot above and set a suitable trial_shift value, then re-run the script with run_trial = 0 and set shift_down_value accordingly.")
    exit(0)  # Stop here after the trial

# Old function removed - replaced by analyze_roi_evolution_direct()

def compute_roi_centroid_fourier_intensity(centroids, roi_polygon, image_shape):
    """
    Given centroids (N,2), ROI polygon, and image shape, create a binary image with 1s at centroid locations inside ROI,
    then crop to the ROI bounding box and perform FFT, returning the intensity map for the ROI region only.
    Returns: (intensity_map, cropped_shape) where cropped_shape is (height, width) of the cropped region.
    """
    from skimage.draw import polygon2mask
    mask = polygon2mask(image_shape, np.array(roi_polygon))
    binary_img = np.zeros(image_shape, dtype=np.float32)
    if centroids is not None and len(centroids) > 0:
        for y, x in centroids:
            if 0 <= int(y) < image_shape[0] and 0 <= int(x) < image_shape[1]:
                if mask[int(y), int(x)]:
                    binary_img[int(y), int(x)] = 1.0
    # Find bounding box of ROI
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None, None
    y1, y2 = y_coords.min(), y_coords.max() + 1
    x1, x2 = x_coords.min(), x_coords.max() + 1
    cropped = binary_img[y1:y2, x1:x2]
    cropped_shape = cropped.shape  # (height, width)
    FT = np.fft.fft2(cropped)
    FT_shift = np.fft.fftshift(FT)
    intensity_map = np.abs(FT_shift)**2
    center_row = intensity_map.shape[0] // 2
    center_col = intensity_map.shape[1] // 2
    intensity_map[center_row, center_col] = 0.0
    return intensity_map, cropped_shape

def pad_intensity_map_to_size(intensity_map, target_height, target_width):
    """
    Pad an intensity map to a target size by adding zeros around it.
    Centers the original intensity map in the padded result.
    NEVER crops - only pads to ensure no data loss.
    The target size should be the maximum size found in the first pass.
    """
    if intensity_map is None:
        return None
    
    current_height, current_width = intensity_map.shape
    
    # If original image is larger than target, this is an error in our logic
    # The target should always be the maximum size found in first pass
    if current_height > target_height or current_width > target_width:
        print(f"ERROR: Image {current_height}x{current_width} is larger than target {target_height}x{target_width}")
        print("This should not happen if first pass correctly found maximum dimensions.")
        # For now, just return the original image to avoid data loss
        return intensity_map
    
    # Calculate padding needed
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)
    
    # Calculate padding for top/bottom and left/right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Pad the intensity map
    padded_map = np.pad(intensity_map, 
                       ((pad_top, pad_bottom), (pad_left, pad_right)), 
                       mode='constant', constant_values=0)
    
    return padded_map

def compute_roi_masked_image_fourier_intensity(cropped_image, roi_polygon):
    """
    Mask the cropped image to the ROI polygon, crop to the ROI bounding box, and compute the FFT intensity map.
    Returns the intensity map for the ROI region only.
    Returns: (intensity_map, cropped_shape) where cropped_shape is (height, width) of the cropped region.
    """
    from skimage.draw import polygon2mask
    mask = polygon2mask(cropped_image.shape, np.array(roi_polygon))
    masked_img = cropped_image * mask
    # Find bounding box of ROI
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None, None
    y1, y2 = y_coords.min(), y_coords.max() + 1
    x1, x2 = x_coords.min(), x_coords.max() + 1
    cropped = masked_img[y1:y2, x1:x2]
    cropped_shape = cropped.shape  # (height, width)
    FT = np.fft.fft2(cropped)
    FT_shift = np.fft.fftshift(FT)
    intensity_map = np.abs(FT_shift)**2
    center_row = intensity_map.shape[0] // 2
    center_col = intensity_map.shape[1] // 2
    intensity_map[center_row, center_col] = 0.0
    return intensity_map, cropped_shape

def compute_roi_average_psi(psi_image, roi_polygon):
    """
    Compute the average ψₙ value within the ROI.
    
    Parameters:
    - psi_image: ψₙ image (2D numpy array)
    - roi_polygon: ROI polygon coordinates [(x1, y1), (x2, y2), ...] in SHIFTED coordinates
    
    Returns:
    - average_psi: Average ψₙ value within ROI or None if failed
    - roi_area: Number of pixels within ROI or None if failed
    """
    try:
        from skimage.draw import polygon2mask
        
        # Convert polygon coordinates to integer indices
        roi_coords = np.array(roi_polygon, dtype=int)
        
        # Create mask for the ROI - ensure we work with 2D coordinates
        if len(psi_image.shape) == 3:
            # For 3D images, create mask for the first 2 dimensions only
            mask = polygon2mask(psi_image.shape[:2], roi_coords)
        else:
            # For 2D images, use the full shape
            mask = polygon2mask(psi_image.shape, roi_coords)
        print(f"    Mask shape: {mask.shape}, True pixels: {np.sum(mask)}")
        
        # Apply mask to image and get non-zero values
        # Handle both 2D and 3D images
        if len(psi_image.shape) == 3:
            # For 3D images, apply mask to each channel and combine
            masked_psi = np.zeros_like(psi_image)
            for channel in range(psi_image.shape[2]):
                masked_psi[:, :, channel] = psi_image[:, :, channel] * mask
            # Get non-zero values from all channels
            roi_values = masked_psi[masked_psi > 0]
        else:
            # For 2D images, apply mask directly
            masked_psi = psi_image * mask
            roi_values = masked_psi[masked_psi > 0]  # Only consider non-zero values
        
        if len(roi_values) == 0:
            print("Warning: No non-zero values found within ROI")
            return None, None
        
        # Calculate average and area
        average_psi = np.mean(roi_values)
        # Calculate actual ROI area from the mask (not from non-zero ψₙ values)
        roi_area = np.sum(mask)  # This gives the true polygon area
        
        return average_psi, roi_area
        
    except Exception as e:
        print(f"Error in compute_roi_average_psi: {e}")
        return None, None


def run_pipeline(folder_path, start_t, end_t, n_value, fourier_rect, fps,
                 psi_value_picture, fourier_intensity_picture, psi_value_video, fourier_intensity_picture_video,
                 psi_value_array, fourier_intensity_array, save_fourier_peaks_data, save_fourier_metrics_csv, top_peaks_count, shift_down_value, separation_of_histogram, create_psi_histogram):
    """
    Complete time-series segmentation analysis pipeline for computing n-fold order parameters (ψₙ) and Fourier analysis.
    
    This pipeline processes HDF5 segmentation files across multiple timepoints to:
    1. Extract and analyze segmented regions
    2. Compute n-fold order parameters (ψₙ) using Voronoi tessellation
    3. Perform Fourier analysis on specified regions
    4. Generate various outputs (images, videos, data arrays, metrics)
    
    INPUT PARAMETERS:
    
    File and Time Parameters:
      folder_path                : Base folder containing input segmentation files and output directories
      start_t, end_t             : Timepoint range to process (inclusive)
    
    Analysis Parameters:
      n_value                    : n-fold value for the order parameter ψₙ (typically 6 for hexagonal order)
      width_cut, height_cut      : Pixel cuts for cropping (left/right and top/bottom edges)
      shift_down_value           : Vertical shift in pixels for ψₙ visualization (wraps around)
    
    Fourier Analysis Parameters:
      fourier_rect               : (x1, x2, y1, y2) rectangle coordinates for Fourier intensity analysis
      top_peaks_count            : Number of top peaks to identify in Fourier analysis
    
    Video and Histogram Parameters:
      fps                        : Frames per second for video output generation
      separation_of_histogram    : Bin size for y-axis histogram of ψₙ values
    
    Output Control Flags - Images and Videos:
      psi_value_picture          : Save ψₙ colored images (1: yes, 0: no)
      fourier_intensity_picture  : Save Fourier intensity images (1: yes, 0: no)
      psi_value_video            : Create video from ψₙ images (1: yes, 0: no)
      fourier_intensity_picture_video: Create video from Fourier images (1: yes, 0: no)
      create_psi_histogram       : Create histogram of ψₙ values along y-axis (1: yes, 0: no)
    
    Output Control Flags - Data Arrays and Files:
      psi_value_array            : Save ψₙ image array data as .npy files (1: yes, 0: no)
      fourier_intensity_array    : Save Fourier intensity array data as .npy files (1: yes, 0: no)
      save_fourier_peaks_data    : Save Fourier peaks data (offsets, intensities, distances) (1: yes, 0: no)
      save_fourier_metrics_csv   : Save Fourier metrics table as CSV file (1: yes, 0: no)
    
    OUTPUTS:
    - Images: ψₙ colored maps, Fourier intensity maps, histograms, reciprocal lattice overlays
    - Videos: Time-series videos of ψₙ and Fourier intensity evolution
    - Data: .npy arrays of ψₙ values and Fourier intensities, peak data, metrics CSV
    - All outputs are organized in subdirectories under the main folder_path
    """
    # -------------------------
    import matplotlib.pyplot as plt
    # Helper Functions (Inner)
    # -------------------------
    @lru_cache(maxsize=128)
    def _compute_vorono_cached(points_tuple):
        """Cached Voronoi computation for repeated point patterns."""
        points = np.array(points_tuple)
        return Voronoi(points, qhull_options="QJ")

    def compute_order_parameter(points, n):
        """
        Compute the n-fold order parameter (ψₙ) for 2D points using a weighted Voronoi construction.
        Optimized with vectorization and caching for better performance.
        """
        if len(points) == 0:
            return np.zeros(0, dtype=complex)
        
        # Early exit for single point
        if len(points) == 1:
            return np.array([0j])
        
        valid_dims = [i for i in range(points.shape[1]) if np.any(points[:, i] != points[0, i])]
        if not valid_dims:
            return np.zeros(points.shape[0], dtype=complex)
        reduced_points = points[:, valid_dims]
        if reduced_points.shape[1] < 2:
            return np.zeros(points.shape[0], dtype=complex)

        # Use cached Voronoi computation if enabled
        if CACHE_VORONOI:
            # Convert numpy array to tuple for caching (numpy arrays are not hashable)
            points_tuple = tuple(map(tuple, reduced_points))
            vor = _compute_vorono_cached(points_tuple)
        else:
            vor = Voronoi(reduced_points, qhull_options="QJ")
        
        N = len(points)
        contributions = np.zeros(N, dtype=complex)
        weights = np.zeros(N, dtype=float)

        # Vectorized computation of ridge contributions
        if USE_VECTORIZATION and len(vor.ridge_points) > 0:
            ridge_points = np.array(vor.ridge_points)
            ridge_vertices = vor.ridge_vertices
            
            # Filter valid ridges (no infinite vertices)
            valid_ridges = [i for i, vertices in enumerate(ridge_vertices) if -1 not in vertices]
            
            if valid_ridges:
                valid_ridge_points = ridge_points[valid_ridges]
                valid_vertices = [ridge_vertices[i] for i in valid_ridges]
                
                # Vectorized computation
                for (i, j), vertices in zip(valid_ridge_points, valid_vertices):
                    v0, v1 = vor.vertices[vertices[0]], vor.vertices[vertices[1]]
                    ridge_length = np.linalg.norm(v0 - v1)
                    weight = ridge_length**2

                    vec_ij = reduced_points[j] - reduced_points[i]
                    theta_ij = np.arctan2(vec_ij[1], vec_ij[0])
                    contributions[i] += weight * np.exp(1j * n * theta_ij)
                    weights[i] += weight

                    theta_ji = np.arctan2(-vec_ij[1], -vec_ij[0])
                    contributions[j] += weight * np.exp(1j * n * theta_ji)
                    weights[j] += weight
        else:
            # Fallback to original method
            for (i, j), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
                if -1 in ridge_vertices:
                    continue
                v0, v1 = vor.vertices[ridge_vertices[0]], vor.vertices[ridge_vertices[1]]
                ridge_length = np.linalg.norm(v0 - v1)
                weight = ridge_length**2

                vec_ij = reduced_points[j] - reduced_points[i]
                theta_ij = np.arctan2(vec_ij[1], vec_ij[0])
                contributions[i] += weight * np.exp(1j * n * theta_ij)
                weights[i] += weight

                theta_ji = np.arctan2(-vec_ij[1], -vec_ij[0])
                contributions[j] += weight * np.exp(1j * n * theta_ji)
                weights[j] += weight

        # Vectorized final computation
        psi_n = np.where(weights > 0, contributions / weights, 0)
        return psi_n

    def get_fourier_intensity_and_metric(image, x1, x2, y1, y2):
        """
        Extract a rectangular region from the input image, compute its 2D Fourier transform,
        zero-out the central (DC) component, and return both the intensity map and a metric.
        Metric: max(intensity)/(mean(intensity)) after DC removal.
        """
        region = image[y1:y2, x1:x2]
        FT = np.fft.fft2(region)
        FT_shift = np.fft.fftshift(FT)
        intensity_map = np.abs(FT_shift)**2
    
        # Zero-out the DC component.
        center_row = intensity_map.shape[0] // 2
        center_col = intensity_map.shape[1] // 2
        intensity_map[center_row, center_col] = 0.0
    
        mean_intensity = np.mean(intensity_map) if np.mean(intensity_map) != 0 else 1.0
        metric = np.max(intensity_map) / mean_intensity
        return intensity_map, metric

    def find_top_peaks_and_relative_locations(intensity_map, num_peaks=10):
        """
        Find the coordinates of the top 'num_peaks' values in the 2D intensity map,
        compute their offsets relative to the center, and calculate Euclidean distances.
        Returns:
          relative_peak_coords : List of (row_offset, col_offset) tuples.
          peak_values          : List of intensity values.
          distances            : List of Euclidean distances.
        Optimized with vectorized operations for better performance.
        """
        # Vectorized peak finding
        flat_indices = np.argsort(intensity_map.ravel())
        top_indices = flat_indices[-num_peaks:]
        top_indices = top_indices[np.argsort(-intensity_map.ravel()[top_indices])]
    
        # Vectorized coordinate extraction
        peak_coords = [np.unravel_index(idx, intensity_map.shape) for idx in top_indices]
        peak_values = [intensity_map[coord] for coord in peak_coords]
    
        # Pre-compute center coordinates
        center_row = intensity_map.shape[0] / 2.0
        center_col = intensity_map.shape[1] / 2.0
    
        # Vectorized offset and distance computation
        if USE_VECTORIZATION:
            peak_coords_array = np.array(peak_coords)
            offsets = peak_coords_array - np.array([center_row, center_col])
            relative_peak_coords = [(offset[0], offset[1]) for offset in offsets]
            distances = np.sqrt(np.sum(offsets**2, axis=1))
        else:
            # Original method
            relative_peak_coords = []
            distances = []
            for coord in peak_coords:
                offset = (coord[0] - center_row, coord[1] - center_col)
                relative_peak_coords.append(offset)
                distances.append(np.sqrt(offset[0]**2 + offset[1]**2))
    
        return relative_peak_coords, peak_values, distances
    

    def process_segmentation_timepoint(t, folder_path, image_dir, n_value,
                                       save_psi_array=False, list_dir='', psi_prefix="", shift_down_value=0,
                                       base_psi_image_dir_shifted=None, psi_prefix_shifted=None, base_psi_list_dir=None,
                                       duplicate_for_boundary=False):
        """
        Process segmentation at a given timepoint:
         - Read the HDF5 file, crop the image, label regions, and compute ψₙ.
         - Color each region based on |ψₙ|.
         - Optionally save the colored image (if image_dir is provided) and
           save the underlying ψ array (if list_dir is provided).
        Returns: data_binary_cropped, chunk_info, psi_array, centroids, psi_n, regions_cropped
        Optimized with vectorization and memory efficiency.
        """
        seg_dir = Path(folder_path)
        pattern = f"*T{t:04d}_Simple Segmentation.h5"
        matches = sorted(seg_dir.glob(pattern))
        if not matches:
            print(f"No segmentation file for T{t:04d} in {seg_dir}")
            return None, None, None, None, None, None
        full_path = matches[0]
        print(f"Processing segmentation file: {full_path}")
        print(f"Processing file: {full_path}")
    
        try:
            with h5py.File(full_path, "r") as f:
                dataset_name = list(f.keys())[0]
                data = np.array(f[dataset_name])
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            return None, None, None, None, None, None
    
        data = np.squeeze(data)
        
        # Duplicate H5 data for periodic boundary conditions if enabled
        original_height, original_width = data.shape
        if duplicate_for_boundary:
            data = np.vstack([data, data, data])  # Stack 3 copies of raw H5 data
            print(f"Duplicated H5 segmentation data for boundary conditions: {(original_height, original_width)} -> {data.shape}")
        else:
            print(f"Using original H5 segmentation data: {data.shape}")
        
        # Vectorized binary conversion
        data_binary = np.where(data == 2, 0, data)  # 2 = background, 1 = cluster
    
        height_img, width_img = data.shape
        data_binary = np.where(data == 2, 0, data)  # 2 = background, 1 = cluster
    
        chunk_info = {
            "original_size": (original_height, data.shape[1]) if duplicate_for_boundary else data.shape,
            "processed_size": data.shape,
            "x_range": (0, width_img),
            "y_range": (0, height_img)
        }
        # Show original image size for user clarity
        if duplicate_for_boundary:
            original_size = (original_height, data.shape[1])
            print(f"Image size: {original_size} (original) + duplicated to {data.shape} for boundary conditions")
        else:
            print(f"Image size: {data.shape}")
        print(f"Processing full image")
        print(f"X-range (columns): {chunk_info['x_range']}")
        print(f"Y-range (rows): {chunk_info['y_range']}")
    
        # Label regions and compute centroids with optimization
        labeled_data = measure.label(data_binary)
        regions = measure.regionprops(labeled_data)
    
        image_shape = (data.shape[0], data.shape[1], 3)
        img_out = np.full(image_shape, 0.5, dtype=np.float32)  # Use float32 for memory efficiency
    
        if len(regions) == 0:
            print(f"No regions found in {full_path}.")
            centroids = np.empty((0, 2))
        else:
            # Vectorized centroid extraction
            centroids = np.array([r.centroid for r in regions])
            
            # FIXED: Compute psi values from ORIGINAL centroids first
            # Don't shift centroids yet - we need original positions for psi computation

        psi_n = compute_order_parameter(centroids, n_value)
        psi_n_magnitude = np.abs(psi_n)
    
        # Pre-compute normalization and colormap for efficiency
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = colormaps["viridis"]
        
        # Create colored image with vectorized operations
        img_out = np.full(image_shape, 0.5, dtype=np.float32)  # Grey background
        
        if centroids.shape[0] > 0:
            # Vectorized coloring of regions
            if USE_VECTORIZATION:
                # Pre-compute colors for all regions
                colors = cmap(norm(psi_n_magnitude))[:, :3]
                
                # Color each region efficiently
                for i, region in enumerate(regions):
                    color = colors[i]
                    coords = region.coords
                    img_out[coords[:, 0], coords[:, 1], :] = color
            else:
                # Original method
                for i, region in enumerate(regions):
                    color = cmap(norm(psi_n_magnitude[i]))[:3]
                    for coord in region.coords:
                        img_out[coord[0], coord[1], :] = color
            
            # Vectorized image shift
            if shift_down_value != 0:
                img_out = np.roll(img_out, shift_down_value, axis=0)
                    
            print(f"Created colored image and shifted by {shift_down_value} pixels")
            
            # FIXED: Now shift centroids for histogram analysis
            if shift_down_value != 0:
                image_height = data.shape[0]
                centroids[:, 0] = (centroids[:, 0] - shift_down_value) % image_height
                print(f"Shifted centroids for histogram analysis")
    
        # Crop image back to original size if boundary duplication was used
        if duplicate_for_boundary:
            # Extract the middle section (original image) from the 3x duplicated image
            start_row = original_height
            end_row = 2 * original_height
            img_out_cropped = img_out[start_row:end_row, :, :]
            print(f"Cropped image from {img_out.shape} to original size {img_out_cropped.shape}")
        else:
            img_out_cropped = img_out
    
        # Save the colored ψₙ image and array only if shift_down_value == 0
        if shift_down_value == 0:
            if image_dir is not None:
                image_dir_str = str(image_dir)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img_out_cropped, extent=[0, img_out_cropped.shape[1], 0, img_out_cropped.shape[0]])
                t_min = (t - start_t) * 0.5
                ax.set_title(f"{t_min:.1f} min - ψ_{n_value} map")
                ax.set_xlabel('X-coordinate (pixels)')
                ax.set_ylabel('Y-coordinate (pixels)')
                ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
                ax.set_xticks(np.arange(0, img_out_cropped.shape[1]+1, 200))
                ax.set_yticks(np.arange(0, img_out_cropped.shape[0]+1, 200))
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.grid(True, alpha=0.3, color='white', linewidth=0.5, which='both')
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
                cbar.set_label(r"$|\psi_%d|$" % n_value)
                image_filename = os.path.join(image_dir_str, f"{psi_prefix}_T{t:04d}_colored.png")
                fig.savefig(image_filename, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"Saved ψₙ image to {image_filename}")
            else:
                plt.close()
            if save_psi_array and list_dir is not None:
                psi_array_filename = os.path.join(str(list_dir), f"{psi_prefix}_psi_array_T{t:04d}.npy")  # type: ignore
                np.save(psi_array_filename, img_out_cropped)
                print(f"Saved ψₙ array to {psi_array_filename}")
        # Save the shifted ψₙ image and array only if shift_down_value != 0 and base_psi_image_dir_shifted is not None:
        if shift_down_value != 0 and base_psi_image_dir_shifted is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_out_cropped, extent=[0, img_out_cropped.shape[1], 0, img_out_cropped.shape[0]])
            t_min = (t - start_t) * 0.5
            ax.set_title(f"{t_min:.1f} min - ψ_{n_value} map (Shifted by {shift_down_value} pixels)")
            ax.set_xlabel('X-coordinate (pixels)')
            ax.set_ylabel('Y-coordinate (pixels)')
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            ax.set_xticks(np.arange(0, img_out_cropped.shape[1]+1, 200))
            ax.set_yticks(np.arange(0, img_out_cropped.shape[0]+1, 200))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5, which='both')
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label(r"$|\psi_%d|$" % n_value)
            shifted_image_filename = os.path.join(base_psi_image_dir_shifted, f"{psi_prefix_shifted}_T{t:04d}_colored.png")
            fig.savefig(shifted_image_filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved shifted ψₙ image to {shifted_image_filename}")
            print(f"DEBUG: save_psi_array = {save_psi_array}, base_psi_list_dir = {base_psi_list_dir}")
            # Save the shifted psi_value array in the shifted list directory
            if save_psi_array and base_psi_list_dir is not None:
                shifted_list_dir = base_psi_list_dir + f'_Shift_{shift_down_value}'
                os.makedirs(shifted_list_dir, exist_ok=True)
                shifted_array_filename = os.path.join(shifted_list_dir, f"{psi_prefix_shifted}_psi_array_T{t:04d}.npy")
                # Apply the shift to the raw psi_value array before saving
                if 'psi_n' in locals() and psi_n is not None:
                    # psi_n is a 2D array (raw values)
                    shifted_psi_n = np.roll(psi_n, shift_down_value, axis=0)
                    np.save(shifted_array_filename, shifted_psi_n)
                    print(f"Saved shifted raw ψₙ array to {shifted_array_filename}")
                else:
                    # Fallback: save the colored image if raw is not available
                    np.save(shifted_array_filename, img_out)
                    print(f"[WARNING] Raw ψₙ array not found, saved colored image array instead to {shifted_array_filename}")
            else:
                print(f"DEBUG: Not saving shifted array because save_psi_array={save_psi_array} or base_psi_list_dir={base_psi_list_dir}")
    
        return data_binary, chunk_info, img_out_cropped, centroids, psi_n, regions

    def process_fourier_timepoint(cropped_image, fourier_rect, image_dir, t, start_t,
                                  save_fourier_array=False, list_dir=None, fourier_prefix="",
                                  num_peaks=10, peaks_list_dir=None):
        """
        Process Fourier analysis on a cropped image:
          - Extract the specified rectangular region.
          - Compute its 2D Fourier transform (with DC removed) to obtain an intensity map.
          - Find the top 'num_peaks' peaks, reporting offsets, intensities, and distances.
          - Optionally save the Fourier intensity image (if image_dir is provided) and array (if list_dir is provided).
          - Also optionally save the Fourier peaks data (if peaks_list_dir is provided).
        Returns: (fourier_metric, peaks_data)
        """
        x1, x2, y1, y2 = fourier_rect
        if x1 < 0 or y1 < 0 or x2 > cropped_image.shape[1] or y2 > cropped_image.shape[0]:
            print("Specified Fourier rectangle is out of bounds!")
            return None, None
    
        intensity_map, fourier_metric = get_fourier_intensity_and_metric(cropped_image, x1, x2, y1, y2)
        relative_peak_coords, peak_values, distances = find_top_peaks_and_relative_locations(intensity_map, num_peaks=num_peaks)
    
        print(f"Timepoint {t} Fourier metric: {fourier_metric}")
        print(f"Top {num_peaks} Fourier peaks (relative offsets from center), intensities, and distances:")
        for rel_coord, value, dist in zip(relative_peak_coords, peak_values, distances):
            print(f"  Relative coordinate {rel_coord}, Intensity: {value:.2e}, Distance: {dist:.2e}")
    
        average_distance = np.mean(distances)
        print(f"Average distance for top {num_peaks} peaks: {average_distance:.2e}")
    
        # Save the Fourier intensity image if image_dir is provided.
        if image_dir is not None:
            fig, ax = plt.subplots(figsize=(8, 8))  # Fixed figure size
            im = ax.imshow(intensity_map, cmap='inferno', extent=[0, intensity_map.shape[1], 0, intensity_map.shape[0]])
            t_min = (t - start_t) * 0.5
            ax.set_title(f"Fourier Intensity Map ({t_min:.1f} min)")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Intensity |FT|²')
            fourier_image_filename = os.path.join(image_dir, f"{fourier_prefix}_T{t:04d}_fourier.png")
            fig.savefig(fourier_image_filename, dpi=300, format='png')
            plt.close(fig)
            print(f"Saved Fourier image to {fourier_image_filename}")
        else:
            plt.close()
    
        # Optionally, save the Fourier intensity array.
        if save_fourier_array and (list_dir is not None):
            fourier_array_filename = os.path.join(list_dir, f"{fourier_prefix}_fourier_array_T{t:04d}.npy")
            np.save(fourier_array_filename, intensity_map)
            print(f"Saved Fourier intensity array to {fourier_array_filename}")
    
        peaks_data = {
            "relative_peak_coords": relative_peak_coords,
            "peak_values": peak_values,
            "distances": distances,
            "average_distance": average_distance
        }
    
        # Optionally, save the Fourier peaks data to its own directory.
        if peaks_list_dir is not None:
            peaks_filename = os.path.join(peaks_list_dir, f"{fourier_prefix}_PeaksData_T{t:04d}_numPeaks_{num_peaks}.npy")
            np.save(peaks_filename, peaks_data)
            print(f"Saved Fourier peaks data to {peaks_filename}")
    
        return fourier_metric, peaks_data

    def create_video_from_images(image_files, video_output_path, fps=2):
        """
        Create a video from a list of image files.
        """
        if not image_files:
            print("No images available for video.")
            return
        writer = imageio.get_writer(video_output_path, fps=fps, format='FFMPEG')
        for img_file in image_files:
            try:
                frame = imageio.imread(img_file)
                writer.append_data(frame)
            except Exception as e:
                print(f"Error reading {img_file}: {e}")
        writer.close()
        print(f"Video saved to {video_output_path}")

    def create_psi_histogram_along_y(centroids, psi_values, cropped_height, separation, image_dir, t, psi_prefix, shift_value=0):
        """
        Create a histogram showing average psi values along the y-axis.
        
        Args:
            centroids: Array of centroid coordinates (AFTER SHIFTING)
            psi_values: Array of psi values for each centroid
            cropped_height: Height of the cropped image
            separation: Bin size for the histogram
            image_dir: Directory to save the histogram
            t: Timepoint
            psi_prefix: Prefix for the filename
            shift_value: Amount of shift applied (for title)
        """
        if len(centroids) == 0:
            print(f"No centroids available for histogram at timepoint {t}")
            return
            
        # Get y-coordinates (AFTER SHIFTING - these are the shifted coordinates)
        y_coords = centroids[:, 0]  # y is the first column after shifting
        psi_magnitudes = np.abs(psi_values)
        
        print(f"Creating histogram for timepoint {t} using SHIFTED centroids")
        print(f"Y-coordinate range after shifting: {np.min(y_coords):.1f} to {np.max(y_coords):.1f}")
        
        # Handle duplicated image height for boundary conditions
        # Use the global duplicate_for_boundary parameter instead of guessing from image size
        if duplicate_for_boundary:
            # If we duplicated the image, the height should be 3x the original
            original_height = cropped_height // 3
            # For histograms, we want to preserve the shift information but map to original range
            # The centroids are already in shifted coordinates, so we map them back to original range
            y_coords_mapped = y_coords % original_height
            effective_height = original_height
            y_coords_for_binning = y_coords_mapped
            print(f"Using duplicated image: mapping {cropped_height} -> {original_height} for histogram (preserving shift)")
        else:
            effective_height = cropped_height
            y_coords_for_binning = y_coords
        
        # Create bins along y-axis
        bins = np.arange(0, effective_height + separation, separation)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate average psi magnitude for each bin
        avg_psi_per_bin = []
        for i in range(len(bins) - 1):
            mask = (y_coords_for_binning >= bins[i]) & (y_coords_for_binning < bins[i + 1])
            if np.any(mask):
                # Calculate average normally
                bin_psi_values = psi_magnitudes[mask]
                avg_psi = np.mean(bin_psi_values)
                print(f"Bin {i}: y={bins[i]}-{bins[i+1]}, count={np.sum(mask)}, avg_psi={avg_psi:.3f}")
                avg_psi_per_bin.append(avg_psi)
            else:
                avg_psi_per_bin.append(0.0)
                print(f"Bin {i}: y={bins[i]}-{bins[i+1]}, count=0, avg_psi=0.0")
        
        # Create the histogram
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, avg_psi_per_bin, width=separation*0.8, alpha=0.7, color='blue')
        plt.xlabel('Y-coordinate (AFTER SHIFTING)')
        plt.ylabel('Average |ψₙ| magnitude')
        # Add shift information to title
        if shift_value != 0:
            t_min = (t - start_t) * 0.5
            plt.title(f'{t_min:.1f} min - Average ψₙ magnitude along Y-axis (Shifted by {shift_value} pixels)')
        else:
            t_min = (t - start_t) * 0.5
            plt.title(f'{t_min:.1f} min - Average ψₙ magnitude along Y-axis (No shift)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        # Save the histogram
        if image_dir is not None:
            histogram_filename = os.path.join(str(image_dir), f"{psi_prefix}_T{t:04d}_y_histogram.png")
            plt.savefig(histogram_filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved ψₙ histogram (using SHIFTED data) to {histogram_filename}")
        else:
            plt.close()

    def create_smoothed_psi_histogram_along_y(centroids, smoothed_psi_values, cropped_height, separation, image_dir, t, smoothed_psi_prefix, shift_value=0, data_dir=None):
        """
        Create a histogram showing average smoothed psi values along the y-axis.
        
        Args:
            centroids: Array of centroid coordinates (AFTER SHIFTING)
            smoothed_psi_values: Array of smoothed psi values for each centroid
            cropped_height: Height of the cropped image
            separation: Bin size for the histogram
            image_dir: Directory to save the histogram
            t: Timepoint
            smoothed_psi_prefix: Prefix for the filename
            shift_value: Amount of shift applied (for title)
        """
        from scipy import stats  # Import at function level for better performance
        
        if len(centroids) == 0:
            print(f"No centroids available for smoothed histogram at timepoint {t}")
            return
            
        # Get y-coordinates (AFTER SHIFTING - these are the shifted coordinates)
        y_coords = centroids[:, 0]  # y is the first column after shifting
        smoothed_psi_magnitudes = smoothed_psi_values  # Already magnitudes from smoothing function
        
        print(f"Creating smoothed histogram for timepoint {t} using SHIFTED centroids")
        print(f"Y-coordinate range after shifting: {np.min(y_coords):.1f} to {np.max(y_coords):.1f}")
        
        # Handle duplicated image height for boundary conditions
        # Use the global duplicate_for_boundary parameter instead of guessing from image size
        if duplicate_for_boundary:
            # If we duplicated the image, the height should be 3x the original
            original_height = cropped_height // 3
            # For histograms, we want to preserve the shift information but map to original range
            # The centroids are already in shifted coordinates, so we map them back to original range
            y_coords_mapped = y_coords % original_height
            effective_height = original_height
            y_coords_for_binning = y_coords_mapped
            print(f"Using duplicated image: mapping {cropped_height} -> {original_height} for smoothed histogram (preserving shift)")
        else:
            effective_height = cropped_height
            y_coords_for_binning = y_coords
        
        # Create bins along y-axis
        bins = np.arange(0, effective_height + separation, separation)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate average smoothed psi magnitude for each bin
        avg_psi_per_bin = []
        bin_names = []
        bin_groups_for_anova = []  # Store individual data points for ANOVA
        
        for i in range(len(bins) - 1):
            mask = (y_coords_for_binning >= bins[i]) & (y_coords_for_binning < bins[i + 1])
            if np.any(mask):
                # Calculate average normally
                bin_psi_values = smoothed_psi_magnitudes[mask]
                avg_psi = np.mean(bin_psi_values)
                print(f"Smoothed Bin {i}: y={bins[i]}-{bins[i+1]}, count={np.sum(mask)}, avg_psi={avg_psi:.3f}")
                avg_psi_per_bin.append(avg_psi)
                bin_groups_for_anova.append(bin_psi_values)  # Store individual values for ANOVA
            else:
                avg_psi_per_bin.append(0.0)
                print(f"Smoothed Bin {i}: y={bins[i]}-{bins[i+1]}, count=0, avg_psi=0.0")
            
                bin_groups_for_anova.append(np.array([]))  # Empty array for empty bins
            
            # Create descriptive bin names with pixel ranges
            bin_names.append(f"bin_{i}_pixels_{bins[i]}-{bins[i+1]}")
        
        # Individual data points already collected above during bin calculation
        
        # Perform statistical tests for independence
        print(f"\n=== Statistical Tests for Independence (T{t:04d}) ===")
        
        # Test 1: Correlation Test (Linear Independence)
        try:
            # Check if we have enough data for correlation test

            if len(y_coords_for_binning) < 3:

                print(f"Correlation Test: Insufficient data ({len(y_coords_for_binning)} points), skipping")

                correlation, p_value_corr = None, None

            else:

                correlation, p_value_corr = stats.pearsonr(y_coords_for_binning, smoothed_psi_magnitudes)
            print(f"Correlation Test:")
            print(f"  Correlation coefficient: {correlation:.4f}")
            print(f"  P-value: {p_value_corr:.6f}")
            print(f"  H₀ (independence): {'Supported' if p_value_corr > 0.05 else 'Rejected'}")
        except Exception as e:
            print(f"Correlation test failed: {e}")
            correlation, p_value_corr = None, None
        
        # Test 2: ANOVA Test (Group Independence)
        try:
            # Filter out empty bins
            non_empty_groups = [group for group in bin_groups_for_anova if len(group) > 0]
            if len(non_empty_groups) >= 2:
                f_stat, p_value_anova = stats.f_oneway(*non_empty_groups)
                print(f"ANOVA Test:")
                print(f"  F-statistic: {f_stat:.4f}")
                print(f"  P-value: {p_value_anova:.6f}")
                print(f"  H₀ (equal means across bins): {'Supported' if p_value_anova > 0.05 else 'Rejected'}")
            else:
                print("ANOVA Test: Insufficient data (need at least 2 non-empty bins)")
                f_stat, p_value_anova = None, None
        except Exception as e:
            print(f"ANOVA test failed: {e}")
            f_stat, p_value_anova = None, None
        
        # Create the histogram
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, avg_psi_per_bin, width=separation*0.8, alpha=0.7, color='green')
        plt.xlabel('Y-coordinate (AFTER SHIFTING)')
        plt.ylabel('Average Smoothed |ψₙ| magnitude')
        # Add shift information to title
        if shift_value != 0:
            t_min = (t - start_t) * 0.5
            plt.title(f'{t_min:.1f} min - Average Smoothed ψₙ magnitude along Y-axis (Shifted by {shift_value} pixels)')
        else:
            t_min = (t - start_t) * 0.5
            plt.title(f'{t_min:.1f} min - Average Smoothed ψₙ magnitude along Y-axis (No shift)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        # Save the histogram
        if image_dir is not None:
            histogram_filename = os.path.join(str(image_dir), f"{smoothed_psi_prefix}_T{t:04d}_y_histogram.png")
            plt.savefig(histogram_filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved smoothed ψₙ histogram (using SHIFTED data) to {histogram_filename}")
            
            # Save histogram data to CSV in the provided data directory
            if data_dir is not None:
                csv_filename = os.path.join(str(data_dir), f"smoothed_histogram_T{t:04d}.csv")
                import csv
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header row
                    header = ['timepoint'] + bin_names
                    writer.writerow(header)
                    # Write data row
                    data_row = [f"T{t:04d}"] + avg_psi_per_bin
                    writer.writerow(data_row)
                print(f"Saved smoothed histogram data to {csv_filename}")
                
                # Save statistical test results
                stats_filename = os.path.join(str(data_dir), f"statistical_tests_T{t:04d}.txt")
                with open(stats_filename, 'w') as f:
                    f.write(f"Statistical Tests for Independence - T{t:04d}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Correlation test results
                    if correlation is not None and p_value_corr is not None:
                        f.write("CORRELATION TEST (Linear Independence):\n")
                        f.write(f"  Correlation coefficient: {correlation:.4f}\n")
                        f.write(f"  P-value: {p_value_corr:.6f}\n")
                        f.write(f"  H₀ (independence): {'Supported' if p_value_corr > 0.05 else 'Rejected'}\n")
                        f.write(f"  Interpretation: {'No linear relationship' if p_value_corr > 0.05 else 'Linear relationship detected'}\n\n")
                    else:
                        f.write("CORRELATION TEST: Failed\n\n")
                    
                    # ANOVA test results
                    if f_stat is not None and p_value_anova is not None:
                        f.write("ANOVA TEST (Group Independence):\n")
                        f.write(f"  F-statistic: {f_stat:.4f}\n")
                        f.write(f"  P-value: {p_value_anova:.6f}\n")
                        f.write(f"  H₀ (equal means across bins): {'Supported' if p_value_anova > 0.05 else 'Rejected'}\n")
                        f.write(f"  Interpretation: {'No significant difference between bins' if p_value_anova > 0.05 else 'Significant differences between bins'}\n\n")
                    else:
                        f.write("ANOVA TEST: Failed or insufficient data\n\n")
                    
                    # Summary
                    f.write("SUMMARY:\n")
                    if correlation is not None and f_stat is not None:
                        independence_supported = (p_value_corr > 0.05) and (p_value_anova > 0.05)
                        f.write(f"  Overall Independence: {'SUPPORTED' if independence_supported else 'REJECTED'}\n")
                        f.write(f"  Evidence: Both tests {'support' if independence_supported else 'reject'} independence hypothesis\n")
                print(f"Saved statistical test results to {stats_filename}")
        else:
            plt.close()
        
        # Return statistical test results for summary analysis
        return {
            'timepoint': t,
            'correlation': correlation,
            'p_value_corr': p_value_corr,
            'f_stat': f_stat,
            'p_value_anova': p_value_anova
        }

    def overlay_reciprocal_lattice_on_psi_image(psi_img, fourier_peaks, fourier_rect, output_path):
        """
        Overlay the reciprocal lattice (from top Fourier peaks) on the psi_value image in the fourier_rect region.
        psi_img: the full RGB psi image (cropped)
        fourier_peaks: dict with 'relative_peak_coords' and 'peak_values' from process_fourier_timepoint
        fourier_rect: (x1, x2, y1, y2)
        output_path: where to save the overlay image
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        x1, x2, y1, y2 = fourier_rect
        region_shape = (y2 - y1, x2 - x1)  # This should be (200, 200)
        
        # Extract the 200x200 region from the psi image
        psi_region = psi_img[y1:y2, x1:x2, :]
        
        # Reconstruct the Fourier domain with only the top peaks
        FT_blank = np.zeros(region_shape, dtype=complex)
        center_row = region_shape[0] // 2
        center_col = region_shape[1] // 2
        
        for (dy, dx), amp in zip(fourier_peaks['relative_peak_coords'], fourier_peaks['peak_values']):
            # Place the amplitude at the peak location (relative to center)
            row = int(round(center_row + dy))
            col = int(round(center_col + dx))
            if 0 <= row < region_shape[0] and 0 <= col < region_shape[1]:
                FT_blank[row, col] = amp
        
        # Inverse FFT to get the lattice
        lattice = np.fft.ifftshift(FT_blank)
        lattice_real = np.abs(np.fft.ifft2(lattice))
        
        # Normalize for display
        lattice_real = lattice_real / np.max(lattice_real) if np.max(lattice_real) > 0 else lattice_real
        
        # Create the overlay image (200x200)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(psi_region)
        
        # Overlay the reciprocal lattice on the 200x200 region
        ax.imshow(lattice_real, cmap='gray', alpha=0.7, vmin=0.5, vmax=1.0)
        
        ax.set_title('Reciprocal Lattice Overlay (200x200 Region)')
        ax.axis('off')
        
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def compute_smoothed_psi_n_magnitude(centroids, psi_n, n_neighbors):
        """
        For each centroid, compute the average |psi_n| of its n_neighbors closest centroids (excluding itself).
        Returns a new array of smoothed |psi_n| values (float).
        """
        if len(centroids) == 0:
            return np.zeros(0, dtype=float)
        if len(centroids) == 1:
            return np.abs(psi_n)
        tree = cKDTree(centroids)
        k = min(n_neighbors+1, len(centroids))
        dists, idxs = tree.query(centroids, k=k)
        smoothed = np.zeros(len(psi_n), dtype=float)
        abs_psi_n = np.abs(psi_n)
        for i in range(len(centroids)):
            neighbor_idxs = idxs[i]
            neighbor_idxs_wo_self = neighbor_idxs[neighbor_idxs != i]
            if len(neighbor_idxs_wo_self) == 0:
                smoothed[i] = abs_psi_n[i]
            else:
                smoothed[i] = np.mean(abs_psi_n[neighbor_idxs_wo_self])
        return smoothed



    # -------------------------
    # Compute Naming Prefixes
    # -------------------------
    psi_prefix = f"Psi_T{start_t}-{end_t}"
    psi_prefix_shifted = f"Psi_T{start_t}-{end_t}_Shift_{shift_down_value}"
    fourier_prefix = f"Fourier_T{start_t}-{end_t}_FourierRect_{fourier_rect[0]}-{fourier_rect[1]}-{fourier_rect[2]}-{fourier_rect[3]}"
    smoothed_psi_prefix = f"PsiSmoothed_T{start_t}-{end_t}_N{n_value}"
    smoothed_psi_prefix_shifted = f"PsiSmoothed_T{start_t}-{end_t}_N{n_value}_Shift_{shift_down_value}"
    
    # ------------------------------------
    # Conditional Directory Creation (Images, Videos, Lists)
    # ------------------------------------
    base_psi_image_dir = None
    base_psi_image_dir_shifted = None
    base_fourier_image_dir = None
    base_psi_histogram_dir = None
    base_reciprocal_lattice_dir = None
    base_smoothed_psi_image_dir = None
    base_smoothed_psi_image_dir_shifted = None
    base_smoothed_psi_histogram_dir = None
    base_smoothed_psi_histogram_dir_shifted = None
    if psi_value_picture or psi_value_video:
        if shift_down_value == 0:
            base_psi_image_dir = os.path.join(folder_path, "image", psi_prefix)
            os.makedirs(base_psi_image_dir, exist_ok=True)
        else:
            base_psi_image_dir_shifted = os.path.join(folder_path, "image", psi_prefix_shifted)
            os.makedirs(base_psi_image_dir_shifted, exist_ok=True)
    if fourier_intensity_picture:
        base_fourier_image_dir = os.path.join(folder_path, "image", fourier_prefix)
        os.makedirs(base_fourier_image_dir, exist_ok=True)
    if create_psi_histogram:
        if shift_down_value == 0:
            base_psi_histogram_dir = os.path.join(folder_path, "image", f"{psi_prefix}_histograms")
        else:
            base_psi_histogram_dir = os.path.join(folder_path, "image", f"{psi_prefix_shifted}_histograms")
        os.makedirs(base_psi_histogram_dir, exist_ok=True)
    if create_reciprocal_lattice_overlay:
        base_reciprocal_lattice_dir = os.path.join(folder_path, "image", f"{fourier_prefix}_reciprocal_lattice_overlay")
        os.makedirs(base_reciprocal_lattice_dir, exist_ok=True)
    if create_smoothed_psi_value_picture:
        if shift_down_value == 0:
            base_smoothed_psi_image_dir = os.path.join(folder_path, "image", smoothed_psi_prefix)
            os.makedirs(base_smoothed_psi_image_dir, exist_ok=True)
        else:
            base_smoothed_psi_image_dir_shifted = os.path.join(folder_path, "image", smoothed_psi_prefix_shifted)
            os.makedirs(base_smoothed_psi_image_dir_shifted, exist_ok=True)
    if create_smoothed_psi_histogram:
        if shift_down_value == 0:
            base_smoothed_psi_histogram_dir = os.path.join(folder_path, "image", f"{smoothed_psi_prefix}_histograms")
            os.makedirs(base_smoothed_psi_histogram_dir, exist_ok=True)
        else:
            base_smoothed_psi_histogram_dir_shifted = os.path.join(folder_path, "image", f"{smoothed_psi_prefix_shifted}_histograms")
            os.makedirs(base_smoothed_psi_histogram_dir_shifted, exist_ok=True)
    
    # Video directories for ψₙ and Fourier intensity videos.
    base_psi_video_dir = None
    base_fourier_video_dir = None
    if psi_value_video:
        base_psi_video_dir = os.path.join(folder_path, "video", psi_prefix)
        os.makedirs(base_psi_video_dir, exist_ok=True)
    if fourier_intensity_picture_video:
        base_fourier_video_dir = os.path.join(folder_path, "video", fourier_prefix)
        os.makedirs(base_fourier_video_dir, exist_ok=True)
    
    # List directories: four separate subdirectories for:
    # (a) ψₙ arrays, (b) Fourier intensity arrays, (c) Fourier peaks data, (d) histogram data.
    base_psi_list_dir = None
    base_fourier_intensity_list_dir = None
    base_fourier_peaks_list_dir = None
    base_histogram_data_list_dir = None
    if psi_value_array:
        base_psi_list_dir = os.path.join(folder_path, "list", "psi_n", psi_prefix)
        os.makedirs(base_psi_list_dir, exist_ok=True)
    if fourier_intensity_array:
        base_fourier_intensity_list_dir = os.path.join(folder_path, "list", "intensity_value", fourier_prefix)
        os.makedirs(base_fourier_intensity_list_dir, exist_ok=True)
    if save_fourier_peaks_data:
        base_fourier_peaks_list_dir = os.path.join(folder_path, "list", "top_peaks", fourier_prefix)
        os.makedirs(base_fourier_peaks_list_dir, exist_ok=True)
    if create_psi_histogram:
        base_histogram_data_list_dir = os.path.join(folder_path, "list", "histogram_data", smoothed_psi_prefix)
        os.makedirs(base_histogram_data_list_dir, exist_ok=True)
    
    # -------------------------
    # Main Processing Loop
    # -------------------------
    segmentation_results = {}
    cropped_images_for_piv = []
    statistical_results = []  # Collect statistical test results from all timepoints
    for t in range(start_t, end_t + 1):
        processed_image, chunk_info, psi_array, centroids, psi_n, regions = process_segmentation_timepoint(
            t, folder_path, base_psi_image_dir, n_value,
            save_psi_array=bool(psi_value_array), list_dir=base_psi_list_dir if base_psi_list_dir is not None else '', psi_prefix=psi_prefix, shift_down_value=shift_down_value,
            base_psi_image_dir_shifted=base_psi_image_dir_shifted, psi_prefix_shifted=psi_prefix_shifted, base_psi_list_dir=base_psi_list_dir,
            duplicate_for_boundary=duplicate_for_boundary
        )
        if processed_image is not None:
            # Store processed image for PIV (apply shift)
            if shift_down_value != 0:
                shifted_processed_image = np.roll(processed_image, shift_down_value, axis=0)
            else:
                shifted_processed_image = np.copy(processed_image)
            cropped_images_for_piv.append((t, shifted_processed_image))
            segmentation_results[t] = {
                "processed_image": processed_image,
                "chunk_info": chunk_info,
                "psi_array": psi_array,
                "centroids": centroids,
                "psi_n": psi_n,
                "regions": regions
            }
            if not piv_only_mode:
                if create_psi_histogram and base_psi_histogram_dir is not None:
                    histogram_prefix = psi_prefix_shifted if shift_down_value != 0 else psi_prefix
                    # Pass the correct height for histogram processing
                    # If using 3x duplication, we need the full duplicated height for proper coordinate mapping
                    histogram_height = processed_image.shape[0]
                    if duplicate_for_boundary:
                        print(f"Histogram: Using duplicated image height {histogram_height} for coordinate mapping")
                    else:
                        print(f"Histogram: Using original image height {histogram_height}")
                    create_psi_histogram_along_y(centroids, psi_n, histogram_height, separation_of_histogram, base_psi_histogram_dir, t, histogram_prefix, shift_value=shift_down_value)
                
                if create_smoothed_psi_value_picture and centroids.shape[0] > 0:
                    smoothed_psi_n_magnitude = compute_smoothed_psi_n_magnitude(centroids, psi_n, n_value)
                    norm = plt.Normalize(vmin=0, vmax=1)
                    cmap = colormaps["viridis"]
                    img_out = np.full((processed_image.shape[0], processed_image.shape[1], 3), 0.5, dtype=np.float32)
                if USE_VECTORIZATION:
                    colors = cmap(norm(smoothed_psi_n_magnitude))[:, :3]
                    for i, region in enumerate(regions):
                        color = colors[i]
                        coords = region.coords
                        img_out[coords[:, 0], coords[:, 1], :] = color
                else:
                    for i, region in enumerate(regions):
                        color = cmap(norm(smoothed_psi_n_magnitude[i]))[:3]
                        for coord in region.coords:
                            img_out[coord[0], coord[1], :] = color
                if shift_down_value != 0:
                    img_out = np.roll(img_out, shift_down_value, axis=0)
                    
                    # Crop smoothed image back to original size if boundary duplication was used
                    if duplicate_for_boundary:
                        # Get original dimensions from the processed image
                        original_height = processed_image.shape[0] // 3
                        start_row = original_height
                        end_row = 2 * original_height
                        img_out_cropped = img_out[start_row:end_row, :, :]
                        print(f"Smoothed ψₙ: Cropped image from {img_out.shape} to original size {img_out_cropped.shape}")
                    else:
                        img_out_cropped = img_out
                else:
                    # No shift, but still need to crop if boundary duplication was used
                    if duplicate_for_boundary:
                        # Get original dimensions from the processed image
                        original_height = processed_image.shape[0] // 3
                        start_row = original_height
                        end_row = 2 * original_height
                        img_out_cropped = img_out[start_row:end_row, :, :]
                        print(f"Smoothed ψₙ: Cropped image from {img_out.shape} to original size {img_out_cropped.shape}")
                    else:
                        img_out_cropped = img_out
                
                # Store the smoothed image in segmentation_results to avoid double generation
                segmentation_results[t]["smoothed_psi_array"] = img_out_cropped
                    
                if shift_down_value == 0 and base_smoothed_psi_image_dir is not None:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_out_cropped, extent=[0, img_out_cropped.shape[1], 0, img_out_cropped.shape[0]])
                    t_min = (t - start_t) * 0.5
                    ax.set_title(f"{t_min:.1f} min - Smoothed |ψ_{n_value}| map (N={n_value}, magnitude avg)")
                    ax.set_xlabel('X-coordinate (pixels)')
                    ax.set_ylabel('Y-coordinate (pixels)')
                    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
                    ax.set_xticks(np.arange(0, img_out_cropped.shape[1]+1, 200))
                    ax.set_yticks(np.arange(0, img_out_cropped.shape[0]+1, 200))
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.grid(True, alpha=0.3, color='white', linewidth=0.5, which='both')
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
                    cbar.set_label(r"$|\psi_{%d}|$ (smoothed, magnitude avg)" % n_value)
                    image_filename = os.path.join(base_smoothed_psi_image_dir, f"{smoothed_psi_prefix}_T{t:04d}_smoothed_colored.png")
                    fig.savefig(image_filename, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    print(f"Saved smoothed ψₙ image to {image_filename}")
                elif shift_down_value != 0 and base_smoothed_psi_image_dir_shifted is not None:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_out_cropped, extent=[0, img_out_cropped.shape[1], 0, img_out_cropped.shape[0]])
                    t_min = (t - start_t) * 0.5
                    ax.set_title(f"{t_min:.1f} min - Smoothed |ψ_{n_value}| map (N={n_value}, magnitude avg, Shifted by {shift_down_value} pixels)")
                    ax.set_xlabel('X-coordinate (pixels)')
                    ax.set_ylabel('Y-coordinate (pixels)')
                    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
                    ax.set_xticks(np.arange(0, img_out_cropped.shape[1]+1, 200))
                    ax.set_yticks(np.arange(0, img_out_cropped.shape[0]+1, 200))
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.grid(True, alpha=0.3, color='white', linewidth=0.5, which='both')
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
                    cbar.set_label(r"$|\psi_{%d}|$ (smoothed, magnitude avg)" % n_value)
                    image_filename = os.path.join(base_smoothed_psi_image_dir_shifted, f"{smoothed_psi_prefix_shifted}_T{t:04d}_smoothed_colored.png")
                    fig.savefig(image_filename, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    print(f"Saved shifted smoothed ψₙ image to {image_filename}")
                if create_smoothed_psi_histogram and (base_smoothed_psi_histogram_dir is not None or base_smoothed_psi_histogram_dir_shifted is not None):
                    smoothed_histogram_prefix = smoothed_psi_prefix_shifted if shift_down_value != 0 else smoothed_psi_prefix
                    smoothed_histogram_dir = base_smoothed_psi_histogram_dir_shifted if shift_down_value != 0 else base_smoothed_psi_histogram_dir
                    # Pass the correct height for smoothed histogram processing
                    # If using 3x duplication, we need the full duplicated height for proper coordinate mapping
                    smoothed_histogram_height = processed_image.shape[0]
                    if duplicate_for_boundary:
                        print(f"Smoothed Histogram: Using duplicated image height {smoothed_histogram_height} for coordinate mapping")
                    else:
                        print(f"Smoothed Histogram: Using original image height {smoothed_histogram_height}")
                    stats_result = create_smoothed_psi_histogram_along_y(centroids, smoothed_psi_n_magnitude, smoothed_histogram_height, separation_of_histogram, smoothed_histogram_dir, t, smoothed_histogram_prefix, shift_value=shift_down_value, data_dir=base_histogram_data_list_dir)

                    if stats_result is not None:
                        statistical_results.append(stats_result)

    # Create ψₙ video if enabled.
    if not piv_only_mode and psi_value_video and base_psi_video_dir is not None:
        # Use the correct image directory based on shift value
        if shift_down_value == 0:
            image_dir_for_video = base_psi_image_dir
            prefix_for_video = psi_prefix
        else:
            image_dir_for_video = base_psi_image_dir_shifted
            prefix_for_video = psi_prefix_shifted
            
        if image_dir_for_video is not None:
            saved_psi_images = [os.path.join(image_dir_for_video, f"{prefix_for_video}_T{t:04d}_colored.png") 
                                for t in segmentation_results]
            psi_video_filename = os.path.join(base_psi_video_dir, f"{prefix_for_video}_video.mp4")
            create_video_from_images(saved_psi_images, psi_video_filename, fps=fps)
        else:
            print("Warning: No image directory available for video creation.")
    
    # Process Fourier analysis for each timepoint.
    if not piv_only_mode:
        fourier_metrics = {}
        for t in segmentation_results:
            processed_image = segmentation_results[t]["processed_image"]
            metric, peaks_data = process_fourier_timepoint(
                processed_image, fourier_rect, base_fourier_image_dir, t, start_t,
                save_fourier_array=bool(fourier_intensity_array),
                list_dir=base_fourier_intensity_list_dir,
                fourier_prefix=fourier_prefix,
                num_peaks=top_peaks_count,
                peaks_list_dir=base_fourier_peaks_list_dir if save_fourier_peaks_data else None
            )
            fourier_metrics[t] = metric
            # Overlay reciprocal lattice if enabled
            if create_reciprocal_lattice_overlay and base_reciprocal_lattice_dir is not None:
                psi_img = segmentation_results[t]["psi_array"]
                output_path = os.path.join(base_reciprocal_lattice_dir, f"{fourier_prefix}_T{t:04d}_reciprocal_lattice_overlay.png")
                overlay_reciprocal_lattice_on_psi_image(psi_img, peaks_data, fourier_rect, output_path)
    else:
        print("Skipping Fourier analysis (PIV-only mode)")
        fourier_metrics = {}
    
    # --- PIV Analysis using MATLAB PIVlab data ---
    if piv_analysis:
        print("Loading PIVlab data from MATLAB file...")
        u_velocities, v_velocities, x_coords, y_coords = load_pivlab_data(piv_storage)
        
        if u_velocities and v_velocities:
            # Direct PIVlab integration - no intermediate files needed
            pass
            
            # PIV quiver plots removed - coordinate shifting issues
            print("PIV quiver plots skipped due to coordinate shifting issues")
        else:
            print("Failed to load PIVlab data. Skipping PIV analysis.")

    # ——— save metrics to CSV? ———
    if not piv_only_mode and save_fourier_metrics_csv:
        # 1) make a data/ subfolder
        data_dir = os.path.join(folder_path, "data")
        os.makedirs(data_dir, exist_ok=True)

        # 2) build a simple table & write it out
        import pandas as pd
        # sort by timepoint
        tps = sorted(fourier_metrics.keys())
        df  = pd.DataFrame({
            "timepoint":      tps,
            "fourier_metric": [fourier_metrics[t] for t in tps],
        })

        csv_fn = os.path.join(data_dir, f"{fourier_prefix}_metrics.csv")
        df.to_csv(csv_fn, index=False)
        print(f"Saved Fourier metrics CSV to {csv_fn}")
    
    # Create Fourier intensity video if enabled.
    if not piv_only_mode and fourier_intensity_picture_video and base_fourier_video_dir is not None:
        saved_fourier_images = []
        for t in range(start_t, end_t + 1):
            if base_fourier_image_dir is not None:
                fourier_filename = os.path.join(base_fourier_image_dir, f"{fourier_prefix}_T{t:04d}_fourier.png")
                if os.path.exists(fourier_filename):
                    saved_fourier_images.append(fourier_filename)
                else:
                    print(f"Warning: {fourier_filename} does not exist.")
            else:
                print(f"Warning: base_fourier_image_dir is None for timepoint {t}")
        fourier_video_filename = os.path.join(base_fourier_video_dir, f"{fourier_prefix}_video.mp4")
        create_video_from_images(saved_fourier_images, fourier_video_filename, fps=fps)
    
    if not piv_only_mode:
        print("Fourier intensity metrics (side peaks only) for each time point:")
    for t, metric in fourier_metrics.items():
        print(f"Timepoint {t}: {metric}")
    else:
        print("Skipping Fourier metrics output (PIV-only mode)")

    # ------------------------------------
    # Statistical Tests Summary Analysis
    # ------------------------------------
    if statistical_results:
        create_statistical_summary(statistical_results, folder_path, start_t, end_t)

    # --- ROI PIV Evolution Plot (call here, do not change the function itself) ---
    # Run ROI analysis when:
    # 1. PIV analysis is enabled AND
    # 2. Either: psi_value_picture is True, OR piv_only_mode is True, OR save_roi_psi_average is True
    if piv_analysis and (psi_value_picture or piv_only_mode or save_roi_psi_average):
        # Show why ROI analysis is running
        reasons = []
        if psi_value_picture:
            reasons.append("psi_value_picture=True")
        if piv_only_mode:
            reasons.append("piv_only_mode=True")
        if save_roi_psi_average:
            reasons.append("save_roi_psi_average=True")
        
        print(f"Starting ROI PIV evolution analysis (enabled by: {', '.join(reasons)})")
        
        # Check if PIV data was successfully loaded
        if not u_velocities or not v_velocities:
            print("Skipping ROI PIV evolution analysis: PIV data not available")
        else:
            print(f"PIV data loaded successfully. Proceeding with ROI analysis...")
            psi_prefix = f"Psi_T{start_t}-{end_t}"
            psi_prefix_shifted = f"Psi_T{start_t}-{end_t}_Shift_{shift_down_value}"
            # For analysis (PIV and FFT), use appropriate directory based on shift value
            # If shift=0, use original directory; if shift≠0, use shifted directory
            if shift_down_value == 0:
                psi_image_dir = os.path.join(folder_path, "image", psi_prefix)  # Use original for no shift
                psi_prefix_used = psi_prefix  # Use original for no shift
            else:
                psi_image_dir = os.path.join(folder_path, "image", psi_prefix_shifted)  # Use shifted for shift≠0
                psi_prefix_used = psi_prefix_shifted  # Use shifted for shift≠0
            # Note: shift_down_value is still passed for ROI coordinate conversion
            
            timepoints = [(t1, t2) for t1, t2 in zip(range(start_t, end_t), range(start_t+1, end_t+1))]
            # Get the image size from the first cropped image used for PIV
            if len(cropped_images_for_piv) > 0:
                image_size = cropped_images_for_piv[0][1].shape
            else:
                image_size = None
            
            # Calculate original height for boundary duplication handling
            if image_size is not None:
                img_height = image_size[0]
                original_height = img_height // 3 if duplicate_for_boundary else img_height
            else:
                original_height = None
            
            # Use direct PIVlab integration (ROI evolution + ROI psi average analysis integrated)
            analyze_roi_evolution_direct(piv_output_dir, roi_polygon, timepoints, 
                                       psi_image_dir, psi_prefix_used, image_size, shift_down_value, 
                                       start_t, end_t, segmentation_results, u_velocities, v_velocities, x_coords, y_coords, original_height)
    else:
        # Show why ROI analysis is NOT running
        reasons = []
        if not piv_analysis:
            reasons.append("piv_analysis=False")
        if not psi_value_picture and not piv_only_mode and not save_roi_psi_average:
            reasons.append("no enabling flags (psi_value_picture, piv_only_mode, or save_roi_psi_average)")
        
        print(f"Skipping ROI PIV evolution analysis (disabled by: {', '.join(reasons)})")

def interpolate_piv_velocity(px, py, u, v, x_coords, y_coords, method='bilinear'):
    """
    Interpolate PIV velocity at a given point using weighted interpolation.
    
    Parameters:
    - px, py: Point coordinates where to interpolate
    - u, v: Velocity field arrays (2D)
    - x_coords, y_coords: Coordinate grids (2D)
    - method: 'bilinear', 'bicubic', 'inverse_distance', or 'nearest'
    
    Returns:
    - u_interp, v_interp: Interpolated velocity components
    """
    from scipy.interpolate import griddata, RegularGridInterpolator
    
    if method == 'bilinear' or method == 'bicubic':
        # Use scipy's griddata for bilinear/bicubic interpolation
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        u_values = u.ravel()
        v_values = v.ravel()
        
        valid_mask = ~(np.isnan(u_values) | np.isnan(v_values))
        if np.sum(valid_mask) == 0:
            # If all values are NaN, return 0
            return 0.0, 0.0
        
        points_valid = points[valid_mask]
        u_values_valid = u_values[valid_mask]
        v_values_valid = v_values[valid_mask]
        
        # Map method names to scipy's expected values
        scipy_method = 'linear' if method == 'bilinear' else 'cubic'
        
        u_interp = griddata(points_valid, u_values_valid, (px, py), method=scipy_method, fill_value=0.0)
        v_interp = griddata(points_valid, v_values_valid, (px, py), method=scipy_method, fill_value=0.0)
        
    elif method == 'inverse_distance':
        # Inverse distance weighting (more robust for irregular grids)
        distances = np.hypot(x_coords - px, y_coords - py)
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        if np.sum(valid_mask) == 0:
            # If all values are NaN, return 0
            return 0.0, 0.0
        
        # Use only valid points
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        distances_valid = distances[valid_mask]
        
        # Avoid division by zero
        distances_valid = np.maximum(distances_valid, 1e-10)
        
        # Use inverse distance weighting (power = 2)
        weights = 1.0 / (distances_valid ** 2)
        
        # Normalize weights
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
            u_interp = np.sum(u_valid * weights)
            v_interp = np.sum(v_valid * weights)
        else:
            u_interp = 0.0
            v_interp = 0.0
    
    else:
        # Fallback to nearest neighbor
        distances = np.hypot(x_coords - px, y_coords - py)
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(u) | np.isnan(v))
        if np.sum(valid_mask) == 0:
            # If all values are NaN, return 0
            return 0.0, 0.0
        
        # Use only valid points
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        distances_valid = distances[valid_mask]
        
        min_idx = np.argmin(distances_valid)
        u_interp = u_valid[min_idx]
        v_interp = v_valid[min_idx]
    
    return u_interp, v_interp

def analyze_roi_evolution_direct(piv_output_dir, roi_polygon, timepoints, psi_image_dir, psi_prefix, image_size, shift_down_value, start_t, end_t, segmentation_results, u_velocities, v_velocities, x_coords, y_coords, original_height):
    """
    Direct version of ROI evolution analysis using PIVlab data without saving intermediate .npy files.
    
    Parameters:
    - piv_output_dir: Output directory for PIV results
    - roi_polygon: ROI polygon coordinates
    - timepoints: List of (t1, t2) timepoint pairs
    - psi_image_dir: Directory containing ψₙ images
    - psi_prefix: Prefix for ψₙ image files
    - image_size: Size of the input images
    - shift_down_value: Y-coordinate shift value
    - start_t, end_t: Start and end timepoints
    - segmentation_results: Segmentation data for each timepoint
    - u_velocities, v_velocities: PIV velocity fields from PIVlab
    - x_coords, y_coords: Coordinate grids from PIVlab
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    import pandas as pd
    
    # Check if PIV data is available
    if not u_velocities or not v_velocities:
        print("PIV data not available. Skipping ROI evolution analysis.")
        return
    
    roi_polygon_str = f"ROI{roi_polygon[0][0]}-{roi_polygon[0][1]}_to_{roi_polygon[2][0]}-{roi_polygon[2][1]}"
    shift_str = f"shift_{shift_down_value}" if shift_down_value != 0 else "no_shift"
    roi_dir = os.path.join(piv_output_dir, f'roi_evolution_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}')
    os.makedirs(roi_dir, exist_ok=True)
    # Create a single directory for all ROI Fourier intensity maps (centroid-based)
    roi_fourier_dir = os.path.join(piv_output_dir, f'roi_fourier_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}')
    os.makedirs(roi_fourier_dir, exist_ok=True)
    # Create a separate directory for masked-image ROI Fourier intensity maps
    roi_fourier_masked_dir = os.path.join(piv_output_dir, f'roi_fourier_masked_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}')
    os.makedirs(roi_fourier_masked_dir, exist_ok=True)
    # Create a directory for masked-image ROI Fourier intensity maps with unfixed intensity (for comparison)
    roi_fourier_masked_unfixed_dir = os.path.join(piv_output_dir, f'roi_fourier_masked_unfixed_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}')
    os.makedirs(roi_fourier_masked_unfixed_dir, exist_ok=True)
    current_polygon = [tuple(pt) for pt in roi_polygon]
    if image_size is None:
        print("Error: No image size available for ROI evolution. Skipping PIV analysis.")
        return
    img_height, img_width = image_size[:2]
    
    # CRITICAL FIX: Calculate original_height for proper coordinate handling
    if duplicate_for_boundary:
        original_height = img_height // 3
        print(f"Image height: {img_height} (tripled), Original height: {original_height}")
    else:
        original_height = img_height
        print(f"Image height: {img_height} (normal)")
    
    # First pass: Find maximum dimensions and intensity values for FFT images
    print("First pass: Finding maximum FFT image dimensions and intensity ranges...")
    max_centroid_height = 0
    max_centroid_width = 0
    max_masked_height = 0
    max_masked_width = 0
    max_centroid_intensity = 0
    max_masked_intensity = 0
    
    # Store evolved polygons for second pass
    evolved_polygons = {}
    
    # Store all dimensions found for debugging and validation
    all_centroid_shapes = []
    all_masked_shapes = []
    
    for t1, t2 in timepoints:
        # Get the corresponding PIV data index
        piv_index = t1 - start_t
        if piv_index < 0 or piv_index >= len(u_velocities):
            t1_min = (t1 - start_t) * 0.5
            t2_min = (t2 - start_t) * 0.5
            print(f"Skipping {t1_min:.1f} min → {t2_min:.1f} min: PIV data not available for this timepoint (index {piv_index}).")
            continue
            
        u = u_velocities[piv_index]
        v = v_velocities[piv_index]
        
        # Create coordinate grid for PIVlab data (for polygon evolution only)
        if x_coords is None or y_coords is None:
            # Create coordinate grid based on velocity field shape (matching process_week7.py)
            y_coords_grid, x_coords_grid = np.meshgrid(
                np.arange(u.shape[0]), 
                np.arange(u.shape[1]), 
                indexing='ij'
            )
        else:
            x_coords_grid = x_coords
            y_coords_grid = y_coords
        
        # Use ORIGINAL coordinate system for PIV (no shifting)
        xy_grid = np.column_stack([x_coords_grid.flatten(), y_coords_grid.flatten()])
        u_flat = u.flatten()
        v_flat = v.flatten()
        
        # Convert shifted ROI points back to original coordinate system for PIV evolution
        if shift_down_value != 0:
            # Unshift the ROI points to original coordinates
            # The current_polygon is in SHIFTED coordinates, need to convert to ORIGINAL coordinates
            unshifted_polygon = []
            for px, py in current_polygon:
                # Convert from shifted coordinates back to original
                # Since shift_down_value is positive (shifts down), we subtract to get original position
                # CRITICAL FIX: Use original_height, not img_height!
                unshifted_py = (py + shift_down_value) % original_height
                unshifted_polygon.append((px, unshifted_py))
            print(f"Converted ROI from shifted coordinates {current_polygon} to original coordinates {unshifted_polygon}")
        else:
            unshifted_polygon = current_polygon
        
        # Evolve polygon in ORIGINAL coordinate system using weighted interpolation
        new_polygon_original = []
        print(f"Using PIV interpolation method: {piv_interpolation_method}")
        for px, py in unshifted_polygon:
            # Use weighted interpolation instead of nearest neighbor for smoother results
            u_interp, v_interp = interpolate_piv_velocity(px, py, u, v, x_coords_grid, y_coords_grid, method=piv_interpolation_method)
            new_px = px + u_interp
            new_py = py + v_interp
            # CRITICAL FIX: Use original_height for boundary clamping
            new_px = max(0, min(new_px, img_width-1))
            new_py = max(0, min(new_py, original_height-1))
            new_polygon_original.append((new_px, new_py))
        
        # Store evolved polygon for second pass
        evolved_polygons[t1] = new_polygon_original
        
        # Check FFT dimensions for this timepoint
        seg_result = segmentation_results.get(t1)
        if seg_result is not None:
            centroids = seg_result['centroids']
            image_shape = seg_result['processed_image'].shape
            processed_image = seg_result['processed_image']
            
            # Convert centroids to original coordinates
            if shift_down_value != 0:
                unshifted_centroids = []
                for y, x in centroids:
                    # Since shift_down_value is positive (shifts down), we subtract to get original position
                    # Use original_height instead of image_shape[0] to keep coordinates in original image space
                    unshifted_y = (y + shift_down_value) % original_height
                    unshifted_centroids.append((unshifted_y, x))
                centroids_for_fft = np.array(unshifted_centroids)
            else:
                centroids_for_fft = centroids
            
            # Check centroid-based FFT dimensions and intensity
            intensity_map, cropped_shape = compute_roi_centroid_fourier_intensity(centroids_for_fft, new_polygon_original, image_shape)
            if cropped_shape is not None:
                all_centroid_shapes.append(cropped_shape)
                max_centroid_height = max(max_centroid_height, cropped_shape[0])
                max_centroid_width = max(max_centroid_width, cropped_shape[1])
                if intensity_map is not None:
                    max_centroid_intensity = max(max_centroid_intensity, np.max(intensity_map))
            
            # Check masked-image FFT dimensions and intensity
            intensity_map_masked, cropped_shape_masked = compute_roi_masked_image_fourier_intensity(processed_image, new_polygon_original)
            if cropped_shape_masked is not None:
                all_masked_shapes.append(cropped_shape_masked)
                max_masked_height = max(max_masked_height, cropped_shape_masked[0])
                max_masked_width = max(max_masked_width, cropped_shape_masked[1])
                if intensity_map_masked is not None:
                    max_masked_intensity = max(max_masked_intensity, np.max(intensity_map_masked))
        
        # Update current polygon for next iteration
        if shift_down_value != 0:
            new_polygon_for_viz = []
            for px, py in new_polygon_original:
                # Convert from original coordinates to shifted coordinates for display
                # Since we want to display on the shifted image, subtract shift_down_value
                # CRITICAL FIX: Use original_height for proper modulo operation
                shifted_py = (py - shift_down_value) % original_height
                new_polygon_for_viz.append((px, shifted_py))
        else:
            new_polygon_for_viz = new_polygon_original
        current_polygon = new_polygon_for_viz
    
    print(f"Maximum FFT dimensions and intensity ranges found:")
    print(f"  Centroid-based: {max_centroid_height} x {max_centroid_width}, max intensity: {max_centroid_intensity:.2e}")
    print(f"  Masked-image: {max_masked_height} x {max_masked_width}, max intensity: {max_masked_intensity:.2e}")
    print(f"  All centroid shapes found: {all_centroid_shapes}")
    print(f"  All masked shapes found: {all_masked_shapes}")
    
    # Set consistent colorbar ranges
    centroid_vmax = max_centroid_intensity
    masked_vmax = max_masked_intensity
    
    print(f"\nROI Fourier Analysis Methods:")
    print(f"  1. Centroid-based: Fixed size + Fixed intensity (for timepoint comparison)")
    print(f"  2. Masked-image: Fixed size + Fixed intensity (for timepoint comparison)")
    print(f"  3. Masked-image (unfixed): Fixed size + Natural intensity (shows actual variations)")
    
    # Second pass: Generate FFT images with consistent size
    print("Second pass: Generating FFT images with consistent size...")
    current_polygon = [tuple(pt) for pt in roi_polygon]  # Reset for second pass
    
    for t1, t2 in timepoints:
        # Get the corresponding PIV data index
        piv_index = t1 - start_t
        if piv_index < 0 or piv_index >= len(u_velocities):
            t1_min = (t1 - start_t) * 0.5
            t2_min = (t2 - start_t) * 0.5
            print(f"Skipping {t1_min:.1f} min → {t2_min:.1f} min: PIV data not available for this timepoint (index {piv_index}).")
            continue
            
        u = u_velocities[piv_index]
        v = v_velocities[piv_index]
        
        psi_img_path = os.path.join(psi_image_dir, f'{psi_prefix}_T{t1:04d}_colored.png')
        if not os.path.exists(psi_img_path):
            t1_min = (t1 - start_t) * 0.5
            t2_min = (t2 - start_t) * 0.5
            print(f"Skipping {t1_min:.1f} min → {t2_min:.1f} min: missing psi image file.")
            continue
            
        print(f"Loaded u shape: {u.shape}, v shape: {v.shape}")
        
        # CRITICAL FIX: Handle NaN values in PIV data
        print(f"PIV velocity ranges: u=[{np.min(u):.2f}, {np.max(u):.2f}], v=[{np.min(v):.2f}, {np.max(v):.2f}]")
        print(f"PIV velocity magnitudes: mean={np.mean(np.sqrt(u**2 + v**2)):.2f}, max={np.max(np.sqrt(u**2 + v**2)):.2f}")

        
        # Create coordinate grid for PIVlab data (for polygon evolution only)
        if x_coords is None or y_coords is None:
            # Create coordinate grid based on velocity field shape (matching process_week7.py)
            y_coords_grid, x_coords_grid = np.meshgrid(
                np.arange(u.shape[0]), 
                np.arange(u.shape[1]), 
                indexing='ij'
            )
        else:
            x_coords_grid = x_coords
            y_coords_grid = y_coords
        
        # Use ORIGINAL coordinate system for PIV (no shifting)
        xy_grid = np.column_stack([x_coords_grid.flatten(), y_coords_grid.flatten()])
        u_flat = u.flatten()
        v_flat = v.flatten()
        
        # Convert shifted ROI points back to original coordinate system for PIV evolution
        if shift_down_value != 0:
            # Unshift the ROI points to original coordinates
            # The current_polygon is in SHIFTED coordinates, need to convert to ORIGINAL coordinates
            unshifted_polygon = []
            for px, py in current_polygon:
                # Convert from shifted coordinates back to original
                # Since shift_down_value is positive (shifts down), we subtract to get original position
                unshifted_py = (py + shift_down_value) % original_height
                unshifted_polygon.append((px, unshifted_py))
            print(f"Converted ROI from shifted coordinates {current_polygon} to original coordinates {unshifted_polygon}")
        else:
            unshifted_polygon = current_polygon
        
        # Evolve polygon in ORIGINAL coordinate system using weighted interpolation
        new_polygon_original = []
        print(f"T{t1:04d}: Evolving polygon from {unshifted_polygon}")
        for i, (px, py) in enumerate(unshifted_polygon):
            # Use weighted interpolation instead of nearest neighbor for smoother results
            u_interp, v_interp = interpolate_piv_velocity(px, py, u, v, x_coords_grid, y_coords_grid, method=piv_interpolation_method)
            new_px = px + u_interp
            new_py = py + v_interp
            # CRITICAL FIX: Use original_height for boundary clamping
            new_px = max(0, min(new_px, img_width-1))
            new_py = max(0, min(new_py, original_height-1))
            new_polygon_original.append((new_px, new_py))
        print(f"T{t1:04d}: Evolved to {new_polygon_original}")
        
        # For FFT analysis, we use ORIGINAL coordinates (same as PIV data)
        # This keeps everything in the same coordinate system for analysis
        new_polygon_for_fft = new_polygon_original  # Use original coordinates for FFT
        
        # Shift the evolved polygon back to match the shifted ψₙ background (for visualization only)
        if shift_down_value != 0:
            new_polygon_for_viz = []
            for px, py in new_polygon_original:
                # Convert from original coordinates to shifted coordinates for display
                # Since we want to display on the shifted image, subtract shift_down_value
                # Use original height for modulo operation when boundary duplication is used
                shifted_py = (py - shift_down_value) % original_height
                new_polygon_for_viz.append((px, shifted_py))
        else:
            new_polygon_for_viz = new_polygon_original
        
        # Plot - use smoothed ψₙ data (already shifted if needed) for cleaner visualization
        seg_result = segmentation_results.get(t1)
        if seg_result is not None and 'smoothed_psi_array' in seg_result and seg_result['smoothed_psi_array'] is not None:
            # Use the smoothed ψₙ array data (already shifted and cropped)
            psi_img = seg_result['smoothed_psi_array']
            print(f"Using smoothed ψₙ array data for T{t1:04d} (already shifted by {shift_down_value} pixels if needed)")
        elif seg_result is not None and 'psi_array' in seg_result and seg_result['psi_array'] is not None:
            # Fallback to regular ψₙ array data if smoothed not available
            psi_img = seg_result['psi_array']
            print(f"Using regular ψₙ array data for T{t1:04d} (smoothed not available)")
        else:
            print(f"[ERROR] No ψₙ data available for T{t1:04d}")
            continue
        
        # Ensure we have the correct image format (RGB, not RGBA)
        if len(psi_img.shape) == 3 and psi_img.shape[2] == 4:
            # Convert RGBA to RGB by removing alpha channel
            psi_img = psi_img[:, :, :3]
            print("Converted RGBA to RGB")
        elif len(psi_img.shape) == 3 and psi_img.shape[2] == 3:
            print("Image is already RGB format")
        else:
            print(f"Unexpected image format: shape {psi_img.shape}")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        # Display the ψₙ background image
        print(f"Image shape: {psi_img.shape}, dtype: {psi_img.dtype}")
        ax.imshow(psi_img, aspect='auto', origin='upper')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, psi_img.shape[1])
        # For plotting, use original height if image is tripled, full height otherwise
        ax.set_ylim(psi_img.shape[0], 0)  # Show full image dimensions
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        poly_patch = MplPolygon(new_polygon_for_viz, closed=True, edgecolor='yellow', fill=False, linewidth=2)
        ax.add_patch(poly_patch)
        shift_info = f" (Shift: {shift_down_value}px)" if shift_down_value != 0 else " (No shift)"
        image_type = "Smoothed ψₙ" if 'smoothed_psi_array' in seg_result and seg_result['smoothed_psi_array'] is not None else "ψₙ"
        # Convert timepoints to minutes (relative to start_t, each step = 0.5 minutes)
        t1_minutes = (t1 - start_t) * 0.5  # start_t = 0 min, start_t+1 = 0.5 min, start_t+2 = 1.0 min, etc.
        t2_minutes = (t2 - start_t) * 0.5
        ax.set_title(f'ROI Evolution: PIV Polygon on {image_type} Image{shift_info}\n{t1_minutes:.1f} min → {t2_minutes:.1f} min')
        out_path = os.path.join(roi_dir, f'roi_on_psi_T{t1:04d}_T{t2:04d}.png')
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'Saved ROI PIV evolution plot to {out_path}')
        print(f"[ROI Polygon] {t1_minutes:.1f} min → {t2_minutes:.1f} min edge points (shifted for visualization):")
        for i, (px, py) in enumerate(new_polygon_for_viz):
            print(f"  Vertex {i}: ({px:.2f}, {py:.2f})")
        
        # --- ROI Fourier Intensity Map (Reciprocal Lattice) ---
        if save_roi_fourier_intensity:
            seg_result = segmentation_results.get(t1)
            if seg_result is not None:
                centroids = seg_result['centroids']
                image_shape = seg_result['processed_image'].shape
                processed_image = seg_result['processed_image']
                
                # Use the stored evolved polygon from first pass
                new_polygon_for_fft = evolved_polygons.get(t1)
                if new_polygon_for_fft is None:
                    print(f"Warning: No evolved polygon found for T{t1:04d}")
                    continue
                
                # Centroid-based ROI FFT - convert centroids to original coordinates for FFT analysis
                if shift_down_value != 0:
                    # Convert shifted centroids back to original coordinates
                    # Since np.roll shifts down by shift_down_value, pixel at y in rolled image was at y - shift_down_value in original
                    # Use original_height instead of image_shape[0] to keep coordinates in original image space
                    unshifted_centroids = []
                    for y, x in centroids:
                        unshifted_y = (y + shift_down_value) % original_height
                        unshifted_centroids.append((unshifted_y, x))
                    centroids_for_fft = np.array(unshifted_centroids)
                else:
                    centroids_for_fft = centroids
                
                # Generate FFT and pad to maximum size
                intensity_map, _ = compute_roi_centroid_fourier_intensity(centroids_for_fft, new_polygon_for_fft, image_shape)
                if intensity_map is not None:
                    # Pad to maximum size
                    padded_intensity_map = pad_intensity_map_to_size(intensity_map, max_centroid_height, max_centroid_width)
                    
                    out_path = os.path.join(roi_fourier_dir, f'roi_fourier_T{t1:04d}.png')
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(padded_intensity_map, cmap='inferno', extent=[0, padded_intensity_map.shape[1], 0, padded_intensity_map.shape[0]], vmin=0, vmax=centroid_vmax)
                    t1_min = (t1 - start_t) * 0.5
                    ax.set_title(f'ROI Fourier Intensity (Centroids)\n{t1_min:.1f} min')
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Intensity |FT|²')
                    fig.savefig(out_path, dpi=150, bbox_inches=None, pad_inches=0)
                    plt.close(fig)
                    print(f'Saved ROI centroid Fourier intensity image to {out_path}')
                
                # Masked-image ROI FFT - uses original coordinates for FFT analysis
                intensity_map_masked, _ = compute_roi_masked_image_fourier_intensity(processed_image, new_polygon_for_fft)
                if intensity_map_masked is not None:
                    # Crop to middle 100x100 pixels
                    h, w = intensity_map_masked.shape
                    center_h, center_w = h // 2, w // 2
                    crop_half = roi_fourier_crop_size // 2
                    
                    # Calculate crop boundaries
                    start_h = max(0, center_h - crop_half)
                    end_h = min(h, center_h + crop_half)
                    start_w = max(0, center_w - crop_half)
                    end_w = min(w, center_w + crop_half)
                    
                    # Crop the intensity map
                    cropped_intensity_map = intensity_map_masked[start_h:end_h, start_w:end_w]
                    
                    out_path_masked = os.path.join(roi_fourier_masked_dir, f'roi_fourier_masked_T{t1:04d}.png')
                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    # Use manual maximum if enabled, otherwise use dynamic maximum
                    vmax_value = roi_fourier_max_value if manual_roi_fourier_max else masked_vmax
                    
                    im = ax.imshow(cropped_intensity_map, cmap='inferno', extent=[0, cropped_intensity_map.shape[1], 0, cropped_intensity_map.shape[0]], vmin=0, vmax=vmax_value)
                    t1_min = (t1 - start_t) * 0.5
                    ax.set_title(f'ROI Fourier Intensity (Masked Image, {roi_fourier_crop_size}x{roi_fourier_crop_size})\n{t1_min:.1f} min')
                    cbar = fig.colorbar(im, ax=ax,
                        orientation='vertical',
                        fraction=0.046,  # adjust width
                        pad=0.04)        # adjust spacing
                    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    cbar.update_ticks()
                    cbar.set_label('Intensity |FT|²')
                    fig.savefig(out_path_masked, dpi=150, bbox_inches=None, pad_inches=0)
                    plt.close(fig)
                    print(f'Saved ROI masked-image Fourier intensity image (cropped {roi_fourier_crop_size}x{roi_fourier_crop_size}) to {out_path_masked}')
                
                # Masked-image ROI FFT with UNFIXED intensity (for comparison) - same size but natural intensity range
                intensity_map_masked_unfixed, _ = compute_roi_masked_image_fourier_intensity(processed_image, new_polygon_for_fft)
                if intensity_map_masked_unfixed is not None:
                    # Pad to maximum size (same as fixed version)
                    padded_intensity_map_masked_unfixed = pad_intensity_map_to_size(intensity_map_masked_unfixed, max_masked_height, max_masked_width)
                    
                    out_path_masked_unfixed = os.path.join(roi_fourier_masked_unfixed_dir, f'roi_fourier_masked_unfixed_T{t1:04d}.png')
                    fig, ax = plt.subplots(figsize=(6, 6))
                    # Use natural intensity range (no vmax constraint)
                    im = ax.imshow(padded_intensity_map_masked_unfixed, cmap='inferno', extent=[0, padded_intensity_map_masked_unfixed.shape[1], 0, padded_intensity_map_masked_unfixed.shape[0]])
                    t1_min = (t1 - start_t) * 0.5
                    ax.set_title(f'ROI Fourier Intensity (Masked Image, Unfixed)\n{t1_min:.1f} min')
                    cbar = fig.colorbar(im, ax=ax,
                        orientation='vertical',
                        fraction=0.046,  # adjust width
                        pad=0.04)        # adjust spacing
                        # force one decimal place on every tick
                    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    cbar.update_ticks()
                    cbar.set_label('Intensity |FT|²')
                    fig.savefig(out_path_masked_unfixed, dpi=150, bbox_inches=None, pad_inches=0)
                    plt.close(fig)
                    print(f'Saved ROI masked-image Fourier intensity image (unfixed) to {out_path_masked_unfixed}')
        current_polygon = new_polygon_for_viz  # Use shifted coordinates for next iteration
    
    # Save ROI boundary points to CSV
    print("Saving ROI boundary points to CSV...")
    roi_boundary_csv_path = os.path.join(piv_output_dir, f'roi_boundary_points_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}.csv')
    
    # Prepare data for CSV: rows = vertex indices, columns = timepoints
    import csv
    vertex_names = [f'Vertex_{i}' for i in range(len(roi_polygon))]
    vertex_names.extend([f'Vertex_{i}_x' for i in range(len(roi_polygon))])
    vertex_names.extend([f'Vertex_{i}_y' for i in range(len(roi_polygon))])
    
    with open(roi_boundary_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row (timepoints)
        header = ['Vertex'] + [f'T{t1:04d}' for t1, t2 in timepoints]
        writer.writerow(header)
        
        # Write x-coordinates for each vertex
        for i in range(len(roi_polygon)):
            row = [f'Vertex_{i}_x']
            for t1, t2 in timepoints:
                if t1 in evolved_polygons:
                    vertex_x = evolved_polygons[t1][i][0]
                    row.append(f'{vertex_x:.2f}')
                else:
                    row.append('N/A')
            writer.writerow(row)
        
        # Write y-coordinates for each vertex
        for i in range(len(roi_polygon)):
            row = [f'Vertex_{i}_y']
            for t1, t2 in timepoints:
                if t1 in evolved_polygons:
                    vertex_y = evolved_polygons[t1][i][1]
                    row.append(f'{vertex_y:.2f}')
                else:
                    row.append('N/A')
            writer.writerow(row)
    
    print(f'Saved ROI boundary points to {roi_boundary_csv_path}')
    
    # Add ROI ψₙ average analysis (if enabled)
    if save_roi_psi_average:
        print(f"\nStarting ROI ψₙ average analysis...")
        
        # Create output directory for ROI ψₙ analysis
        roi_psi_dir = os.path.join(piv_output_dir, f'roi_psi_average_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}')
        os.makedirs(roi_psi_dir, exist_ok=True)
        
        # Store results for plotting and CSV
        timepoint_list = []
        average_psi_list = []
        roi_area_list = []
        
        print(f"Analyzing ROI average ψₙ values for {len(timepoints)} timepoints...")
        
        for t1, t2 in timepoints:
            print(f"Processing T{t1:04d}...")
            
            # Get segmentation result for this timepoint
            seg_result = segmentation_results.get(t1)
            if seg_result is None:
                print(f"  Skipping T{t1:04d}: No segmentation data available")
                continue
            
            # Get the ψₙ array data (already shifted if needed)
            if 'psi_array' in seg_result and seg_result['psi_array'] is not None:
                psi_img = seg_result['psi_array']
                print(f"  Using ψₙ array data for T{t1:04d} (already shifted by {shift_down_value} pixels if needed)")
            else:
                print(f"  [ERROR] No ψₙ data available for T{t1:04d}")
                continue
            
            # Compute average ψₙ within ROI using evolved coordinates if available
            if evolved_polygons and t1 in evolved_polygons:
                # Use evolved ROI coordinates (convert from original to shifted for analysis)
                evolved_polygon_original = evolved_polygons[t1]
                if shift_down_value != 0:
                    # Convert evolved polygon from original to shifted coordinates
                    # Since we want to analyze on the shifted image, convert from original to shifted coordinates
                    evolved_polygon_shifted = []
                    for px, py in evolved_polygon_original:
                        # Use original height for modulo operation when boundary duplication is used
                        shifted_py = (py - shift_down_value) % original_height
                        evolved_polygon_shifted.append((px, shifted_py))
                    roi_polygon_for_analysis = evolved_polygon_shifted
                    print(f"  Using EVOLVED ROI coordinates (shifted): {len(roi_polygon_for_analysis)} vertices")
                else:
                    roi_polygon_for_analysis = evolved_polygon_original
                    print(f"  Using EVOLVED ROI coordinates (original): {len(roi_polygon_for_analysis)} vertices")
                
                print(f"    ψₙ image shape: {psi_img.shape}")
                average_psi, roi_area = compute_roi_average_psi(psi_img, roi_polygon_for_analysis)
                
                if average_psi is not None:
                    timepoint_list.append(t1)
                    average_psi_list.append(average_psi)
                    roi_area_list.append(roi_area)
                    print(f"    T{t1:04d}: Average ψₙ = {average_psi:.4f}, ROI area = {roi_area} pixels")
                else:
                    print(f"    T{t1:04d}: Failed to compute average ψₙ")
            else:
                # Fall back to initial ROI coordinates
                roi_polygon_for_analysis = roi_polygon
                print(f"  Using INITIAL ROI coordinates: {roi_polygon_for_analysis}")
            
            print(f"  ψₙ image shape: {psi_img.shape}")
            average_psi, roi_area = compute_roi_average_psi(psi_img, roi_polygon_for_analysis)
            
            if average_psi is not None:
                timepoint_list.append(t1)
                average_psi_list.append(average_psi)
                roi_area_list.append(roi_area)
                print(f"  T{t1:04d}: Average ψₙ = {average_psi:.4f}, ROI area = {roi_area} pixels")
            else:
                print(f"  T{t1:04d}: Failed to compute average ψₙ")
        
        if len(timepoint_list) > 0:
            # Save data as CSV
            csv_data = {
                'Timepoint': timepoint_list,
                'Average_Psi': average_psi_list,
                'ROI_Area_Pixels': roi_area_list
            }
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(roi_psi_dir, f'roi_psi_average_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved ROI ψₙ data to {csv_path}")
            
            # Create time series plot
            plt.figure(figsize=(10, 6))
            plt.plot(timepoint_list, average_psi_list, 'bo-', linewidth=2, markersize=6)
            plt.xlabel('Timepoint')
            plt.ylabel('Average ψₙ Value')
            plt.title(f'ROI Average ψₙ Values Over Time\n{roi_polygon_str} {shift_str}')
            plt.grid(True, alpha=0.3)
            
            # Smart x-axis labeling to avoid crowding
            num_timepoints = len(timepoint_list)
            if num_timepoints <= 20:
                # For few timepoints, show all
                plt.xticks(timepoint_list)
            elif num_timepoints <= 50:
                # For moderate timepoints, show every 5th
                step = max(1, num_timepoints // 10)
                tick_positions = timepoint_list[::step]
                plt.xticks(tick_positions)
            else:
                # For many timepoints, show every 10th or 20th
                step = max(1, num_timepoints // 15)
                tick_positions = timepoint_list[::step]
                plt.xticks(tick_positions)
                # Add first and last timepoints if not already included
                if timepoint_list[0] not in tick_positions:
                    tick_positions = [timepoint_list[0]] + list(tick_positions)
                if timepoint_list[-1] not in tick_positions:
                    tick_positions = list(tick_positions) + [timepoint_list[-1]]
                plt.xticks(sorted(tick_positions))
            
            # Add ROI area as secondary y-axis
            ax2 = plt.twinx()
            ax2.plot(timepoint_list, roi_area_list, 'rs--', linewidth=1, markersize=4, alpha=0.7)
            ax2.set_ylabel('ROI Area (Pixels)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.tight_layout()
            plot_path = os.path.join(roi_psi_dir, f'roi_psi_average_{shift_str}_T{start_t}-{end_t}_{roi_polygon_str}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved ROI ψₙ plot to {plot_path}")
            
            # Print summary statistics
            print(f"\nROI ψₙ Analysis Summary:")
            print(f"  Timepoints analyzed: {len(timepoint_list)}")
            print(f"  Average ψₙ range: {min(average_psi_list):.4f} to {max(average_psi_list):.4f}")
            print(f"  ROI area range: {min(roi_area_list)} to {max(roi_area_list)} pixels")
            print(f"  Mean ψₙ: {np.mean(average_psi_list):.4f}")
            print(f"  Std ψₙ: {np.std(average_psi_list):.4f}")
    else:
        print("ROI ψₙ average analysis disabled (save_roi_psi_average = False)")
    
    # No need to return evolved_polygons since ROI psi average analysis is now integrated
    print("ROI evolution and ψₙ average analysis completed.")
    
    # ------------------------------------
    # Statistical Tests Summary Analysis
    # ------------------------------------
    # Note: Function defined outside main pipeline for proper scope
# ------------------------------------
# Statistical Tests Summary Analysis Function
# ------------------------------------
def create_statistical_summary(statistical_results, folder_path, start_t, end_t):
    """
    Create summary analysis of statistical tests across all timepoints.
    """
    if not statistical_results:
        print("No statistical results available for summary analysis")
        return
    
    print("\n=== Statistical Tests Summary Analysis ===")
    
    # Extract data
    timepoints = [r['timepoint'] for r in statistical_results]
    correlations = [r['correlation'] for r in statistical_results if r['correlation'] is not None]
    p_values_corr = [r['p_value_corr'] for r in statistical_results if r['p_value_corr'] is not None]
    f_stats = [r['f_stat'] for r in statistical_results if r['f_stat'] is not None]
    p_values_anova = [r['p_value_anova'] for r in statistical_results if r['p_value_anova'] is not None]
    
    # Create summary directory
    summary_dir = os.path.join(folder_path, "image", "statistical_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save p-values to CSV
    csv_filename = os.path.join(summary_dir, f"statistical_tests_summary_T{start_t:04d}-{end_t:04d}.csv")
    import csv
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timepoint', 'correlation', 'p_value_correlation', 'f_statistic', 'p_value_anova', 'h0_supported_corr', 'h0_supported_anova'])
        for r in statistical_results:
            h0_corr = r['p_value_corr'] > 0.05 if r['p_value_corr'] is not None else None
            h0_anova = r['p_value_anova'] > 0.05 if r['p_value_anova'] is not None else None
            writer.writerow([
                r['timepoint'],
                r['correlation'] if r['correlation'] is not None else '',
                r['p_value_corr'] if r['p_value_corr'] is not None else '',
                r['f_stat'] if r['f_stat'] is not None else '',
                r['p_value_anova'] if r['p_value_anova'] is not None else '',
                h0_corr,
                h0_anova
            ])
    print(f"Saved statistical summary CSV to {csv_filename}")
    
    # Create improved combined p-values plot
    plt.figure(figsize=(16, 10))
    
    # Create subplots for better distinction
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # Plot correlation p-values (top subplot)
    valid_corr_data = [(r['timepoint'], r['p_value_corr']) for r in statistical_results if r['p_value_corr'] is not None]
    if valid_corr_data:
        corr_timepoints, corr_p_values = zip(*valid_corr_data)
        
        # Set dynamic minimum threshold - use the actual minimum p-value in the data
        min_plot_threshold = min(p for p in corr_p_values if p > 0)
        corr_p_values_plot = [max(p, min_plot_threshold) for p in corr_p_values]
        
        # Use different colors for significance
        corr_colors = ['#2E8B57' if p > 0.05 else '#DC143C' for p in corr_p_values]  # SeaGreen for H0 supported, Crimson for rejected
        corr_sizes = [80 if p > 0.05 else 100 for p in corr_p_values]  # Larger markers for rejected H0
        
        ax1.scatter(corr_timepoints, corr_p_values_plot, c=corr_colors, s=corr_sizes, alpha=0.8, 
                   edgecolors='white', linewidth=1.5, marker='o', label='Correlation Test')
        ax1.plot(corr_timepoints, corr_p_values_plot, color='#4169E1', alpha=0.4, linewidth=2, linestyle='-')
        
        # Add significance threshold
        ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.8, linewidth=2, label='α = 0.05')
        
        # Formatting for correlation plot
        ax1.set_ylabel('P-value (Log Scale)', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set y-axis limits - use fixed threshold for very small values
        max_p_corr = max(p for p in corr_p_values if p > 0)
        ax1.set_ylim(min_plot_threshold * 0.1, max(max_p_corr * 10, 1))
        ax1.set_title('Correlation Test: Linear Independence Between ψ Values and Y-Coordinates', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add legend for correlation plot
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add significance regions (use fixed threshold for very small values)
        ax1.axhspan(min_plot_threshold * 0.1, 0.05, alpha=0.1, color='red', label='Significant (H₀ rejected)')
        ax1.axhspan(0.05, max_p_corr * 10, alpha=0.1, color='green', label='Not significant (H₀ supported)')
        
        # Add annotation for very small p-values
        very_small_corr = [p for p in corr_p_values if p < min_plot_threshold]
        if very_small_corr:
            ax1.text(0.98, 0.02, f'Note: {len(very_small_corr)} p-values < {min_plot_threshold:.0e}\n(plotted at bottom edge)', 
                    transform=ax1.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot ANOVA p-values (bottom subplot)
    valid_anova_data = [(r['timepoint'], r['p_value_anova']) for r in statistical_results if r['p_value_anova'] is not None]
    if valid_anova_data:
        anova_timepoints, anova_p_values = zip(*valid_anova_data)
        
        # Set dynamic minimum threshold - use the actual minimum p-value in the data
        min_plot_threshold = min(p for p in anova_p_values if p > 0)
        anova_p_values_plot = [max(p, min_plot_threshold) for p in anova_p_values]
        
        # Use different colors and markers for ANOVA
        anova_colors = ['#32CD32' if p > 0.05 else '#FF4500' for p in anova_p_values]  # LimeGreen for H0 supported, OrangeRed for rejected
        anova_sizes = [80 if p > 0.05 else 100 for p in anova_p_values]
        
        ax2.scatter(anova_timepoints, anova_p_values_plot, c=anova_colors, s=anova_sizes, alpha=0.8, 
                   edgecolors='white', linewidth=1.5, marker='s', label='ANOVA Test')
        ax2.plot(anova_timepoints, anova_p_values_plot, color='#8A2BE2', alpha=0.4, linewidth=2, linestyle='-')
        
        # Add significance threshold
        ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.8, linewidth=2, label='α = 0.05')
        
        # Formatting for ANOVA plot
        ax2.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
        ax2.set_ylabel('P-value (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set y-axis limits - use fixed threshold for very small values
        max_p_anova = max(p for p in anova_p_values if p > 0)
        ax2.set_ylim(min_plot_threshold * 0.1, max(max_p_anova * 10, 1))
        ax2.set_title('ANOVA Test: Group Independence Across Y-Coordinate Bins', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add legend for ANOVA plot
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add significance regions (use fixed threshold for very small values)
        ax2.axhspan(min_plot_threshold * 0.1, 0.05, alpha=0.1, color='red', label='Significant (H₀ rejected)')
        ax2.axhspan(0.05, max_p_anova * 10, alpha=0.1, color='green', label='Not significant (H₀ supported)')
        
        # Add annotation for very small p-values (ANOVA typically has extremely small p-values)
        very_small_anova = [p for p in anova_p_values if p < min_plot_threshold]
        if very_small_anova:
            ax2.text(0.98, 0.02, f'Note: {len(very_small_anova)} p-values < {min_plot_threshold:.0e}\n(plotted at bottom edge)', 
                    transform=ax2.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Overall title
    fig.suptitle('Statistical Tests for Independence: ψ Values vs Y-Coordinates\n' + 
                f'Analysis Period: T{start_t:04d} to T{end_t:04d} ({len(statistical_results)} timepoints)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.3)
    
    # Save the plot
    plot_filename = os.path.join(summary_dir, f"statistical_tests_combined_T{start_t:04d}-{end_t:04d}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined statistical tests plot to {plot_filename}")
    
    # Print detailed summary statistics
    if p_values_corr and p_values_anova:
        corr_h0_supported = sum(1 for p in p_values_corr if p > 0.05)
        anova_h0_supported = sum(1 for p in p_values_anova if p > 0.05)
        corr_h0_rejected = len(p_values_corr) - corr_h0_supported
        anova_h0_rejected = len(p_values_anova) - anova_h0_supported
        
        print(f"\n" + "="*80)
        print(f"📊 STATISTICAL TESTS SUMMARY REPORT")
        print(f"="*80)
        print(f"📈 Analysis Period: T{start_t:04d} to T{end_t:04d} ({len(statistical_results)} timepoints)")
        print(f"")
        print(f"🔍 CORRELATION TEST (Linear Independence):")
        print(f"   • Total valid tests: {len(p_values_corr)}")
        print(f"   • H₀ supported (p > 0.05): {corr_h0_supported} ({corr_h0_supported/len(p_values_corr)*100:.1f}%)")
        print(f"   • H₀ rejected (p ≤ 0.05): {corr_h0_rejected} ({corr_h0_rejected/len(p_values_corr)*100:.1f}%)")
        print(f"   • Mean correlation coefficient: {np.mean(correlations):.4f}")
        print(f"   • Mean p-value: {np.mean(p_values_corr):.4f}")
        print(f"   • P-value range: {np.min(p_values_corr):.6f} - {np.max(p_values_corr):.6f}")
        print(f"")
        print(f"📊 ANOVA TEST (Group Independence):")
        print(f"   • Total valid tests: {len(p_values_anova)}")
        print(f"   • H₀ supported (p > 0.05): {anova_h0_supported} ({anova_h0_supported/len(p_values_anova)*100:.1f}%)")
        print(f"   • H₀ rejected (p ≤ 0.05): {anova_h0_rejected} ({anova_h0_rejected/len(p_values_anova)*100:.1f}%)")
        print(f"   • Mean F-statistic: {np.mean(f_stats):.4f}")
        print(f"   • Mean p-value: {np.mean(p_values_anova):.4f}")
        print(f"   • P-value range: {np.min(p_values_anova):.6f} - {np.max(p_values_anova):.6f}")
        print(f"")
        print(f"🎯 OVERALL INTERPRETATION:")
        if corr_h0_supported > corr_h0_rejected:
            print(f"   ✅ Correlation Test: Evidence for LINEAR INDEPENDENCE (no linear relationship)")
        else:
            print(f"   ❌ Correlation Test: Evidence for LINEAR DEPENDENCE (linear relationship detected)")
        
        if anova_h0_supported > anova_h0_rejected:
            print(f"   ✅ ANOVA Test: Evidence for GROUP INDEPENDENCE (uniform distribution)")
        else:
            print(f"   ❌ ANOVA Test: Evidence for GROUP DEPENDENCE (non-uniform distribution)")
        print(f"="*80)
    
    print("Statistical summary analysis completed.")

# ------------------------------------
# Run the Pipeline (only when executed directly)
# ------------------------------------
def main():
    run_pipeline(
        folder_path,
        start_t,
        end_t,
        n_value,
        fourier_rect,
        fps,
        psi_value_picture,
        fourier_intensity_picture,
        psi_value_video,
        fourier_intensity_picture_video,
        psi_value_array,
        fourier_intensity_array,
        save_fourier_peaks_data,
        save_fourier_metrics_csv,
        top_peaks_count,
        shift_down_value,
        separation_of_histogram,
        create_psi_histogram,
    )


if __name__ == "__main__":
    main()
