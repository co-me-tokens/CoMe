"""File parsing and I/O utilities for DTU dataset.

This module provides clean utilities for parsing filenames, reading calibration files,
and managing file paths in the DTU dataset structure.
"""

import os
import re
import glob
import numpy as np
from typing import List, Dict, Set, Optional, Tuple


def parse_view_id(file_path: str) -> int:
    """Extract numeric view ID from DTU filename.
    
    Args:
        file_path: Path to DTU file (image or calibration)
        
    Returns:
        view_id: Numeric view identifier
        
    Raises:
        ValueError: If no numeric ID can be extracted from filename
        
    Examples:
        >>> parse_view_id("/path/to/rect_001_3_r5000.png") 
        1
        >>> parse_view_id("00000049_cam.txt")
        49
    """
    basename = os.path.basename(file_path)
    
    # Look for first sequence of digits in filename
    match = re.search(r'(\d+)', basename)
    if not match:
        raise ValueError(f"Cannot extract view ID from filename: {basename}")
    
    return int(match.group(1))


def read_projection_matrix(calib_file: str) -> np.ndarray:
    """Read 3x4 projection matrix from DTU calibration file.
    
    Args:
        calib_file: Path to .txt calibration file
        
    Returns:
        P: 3x4 projection matrix as float64 array
        
    Raises:
        ValueError: If insufficient values found in file
        FileNotFoundError: If calibration file doesn't exist
        
    Note:
        DTU calibration files contain projection matrix values in text format.
        This function extracts all floating point numbers and reshapes to 3x4.
    """
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    
    # Read file and extract all floating point numbers
    float_values = []
    with open(calib_file, 'r') as f:
        content = f.read()
        
        # Regex pattern for floating point numbers (including scientific notation)
        float_pattern = r'[-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?)'
        matches = re.findall(float_pattern, content)
        float_values = [float(x) for x in matches]
    
    # Check we have enough values for 3x4 matrix
    if len(float_values) < 12:
        raise ValueError(f"Expected at least 12 values for 3x4 matrix, found {len(float_values)} in {calib_file}")
    
    # Take first 12 values and reshape
    matrix_values = float_values[:12]
    P = np.array(matrix_values, dtype=np.float64).reshape(3, 4)
    
    return P


def find_matching_files(image_dir: str, calib_dir: str, 
                       lighting_filter: Optional[str] = None,
                       image_extensions: List[str] = None) -> Tuple[List[str], List[str], List[int]]:
    """Find matching image and calibration files by view ID.
    
    Args:
        image_dir: Directory containing DTU images
        calib_dir: Directory containing DTU calibration files  
        lighting_filter: Filter images by substring (e.g., "_3_" for lighting condition 3)
        image_extensions: List of image extensions to consider (default: common formats)
        
    Returns:
        image_paths: List of matched image file paths
        calib_paths: List of matched calibration file paths  
        view_ids: List of corresponding view IDs
        
    Raises:
        RuntimeError: If no matching files found
    """
    if image_extensions is None:
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(image_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    # Apply lighting filter if specified
    if lighting_filter:
        image_files = [f for f in image_files if lighting_filter in os.path.basename(f)]
    
    if not image_files:
        filter_msg = f" with lighting filter '{lighting_filter}'" if lighting_filter else ""
        raise RuntimeError(f"No image files found in {image_dir}{filter_msg}")
    
    # Find calibration files
    calib_files = glob.glob(os.path.join(calib_dir, "*.txt"))
    if not calib_files:
        raise RuntimeError(f"No calibration files found in {calib_dir}")
    
    # Parse view IDs
    image_dict = {}
    for img_path in image_files:
        try:
            view_id = parse_view_id(img_path)
            image_dict[view_id] = img_path
        except ValueError:
            continue  # Skip files we can't parse
    
    calib_dict = {}
    for calib_path in calib_files:
        try:
            view_id = parse_view_id(calib_path)
            calib_dict[view_id] = calib_path
        except ValueError:
            continue  # Skip files we can't parse
    
    # Find common view IDs
    common_ids = sorted(set(image_dict.keys()) & set(calib_dict.keys()))
    if not common_ids:
        raise RuntimeError("No matching view IDs found between images and calibration files")
    
    # Build matched lists
    matched_images = [image_dict[vid] for vid in common_ids]
    matched_calibs = [calib_dict[vid] for vid in common_ids]
    
    return matched_images, matched_calibs, common_ids


def validate_dtu_structure(root_dir: str, scan_id: int, use_rectified: bool = True) -> Dict[str, str]:
    """Validate DTU dataset directory structure and return paths.
    
    Args:
        root_dir: Root directory of DTU dataset
        scan_id: Scan ID to validate
        use_rectified: Whether to use rectified images
        
    Returns:
        paths: Dictionary with validated directory paths
        
    Raises:
        FileNotFoundError: If required directories are missing
    """
    paths = {}
    
    # Check root directory
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"DTU root directory not found: {root_dir}")
    paths['root'] = root_dir
    
    # Check image directory  
    image_subdir = "Rectified" if use_rectified else "Cleaned"
    image_dir = os.path.join(root_dir, image_subdir, f"scan{scan_id}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths['images'] = image_dir
    
    # Check calibration directory
    calib_dir = os.path.join(root_dir, "Calibration", "cal18")
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
    paths['calibrations'] = calib_dir
    
    # Check points directory (optional - may not exist for all scans)
    points_dir = os.path.join(root_dir, "Points")
    if os.path.isdir(points_dir):
        paths['points'] = points_dir
    
    return paths


def get_available_scans(root_dir: str, use_rectified: bool = True) -> List[int]:
    """Get list of available scan IDs in DTU dataset.
    
    Args:
        root_dir: Root directory of DTU dataset  
        use_rectified: Whether to look for rectified images
        
    Returns:
        scan_ids: List of available scan IDs
    """
    image_subdir = "Rectified" if use_rectified else "Cleaned"
    image_base_dir = os.path.join(root_dir, image_subdir)
    
    if not os.path.isdir(image_base_dir):
        return []
    
    scan_ids = []
    for item in os.listdir(image_base_dir):
        if item.startswith("scan") and os.path.isdir(os.path.join(image_base_dir, item)):
            try:
                scan_id = int(item[4:])  # Extract number after "scan"
                scan_ids.append(scan_id)
            except ValueError:
                continue
    
    return sorted(scan_ids)


def get_lighting_conditions(image_dir: str) -> List[str]:
    """Extract available lighting conditions from image filenames.
    
    Args:
        image_dir: Directory containing DTU images
        
    Returns:
        conditions: List of unique lighting condition substrings found
        
    Example:
        For files like "rect_001_3_r5000.png", "rect_001_7_r5000.png"
        returns ["_3_", "_7_"]
    """
    if not os.path.isdir(image_dir):
        return []
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Extract lighting patterns
    lighting_conditions = set()
    for img_path in image_files:
        basename = os.path.basename(img_path)
        # Look for pattern like "_N_" where N is digits
        matches = re.findall(r'_(\d+)_', basename)
        for match in matches:
            lighting_conditions.add(f"_{match}_")
    
    return sorted(list(lighting_conditions))


def get_point_cloud_files(points_dir: str, scan_id: int) -> Dict[str, str]:
    """Find available point cloud files for a given scan.
    
    Args:
        points_dir: Base points directory
        scan_id: Scan ID
        
    Returns:
        point_clouds: Dictionary mapping method name to PLY file path
        
    Example:
        {"stl": "/path/to/stl/stl001_total.ply", 
         "mvsnet": "/path/to/mvsnet/mvsnet001_total.ply"}
    """
    point_clouds = {}
    
    if not os.path.isdir(points_dir):
        return point_clouds
    
    # Check different reconstruction methods
    for method_dir in os.listdir(points_dir):
        method_path = os.path.join(points_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
            
        # Look for PLY file with naming convention: {method}{scan_id:03d}_total.ply
        ply_pattern = f"{method_dir}{scan_id:03d}_total.ply"
        ply_file = os.path.join(method_path, ply_pattern)
        
        if os.path.exists(ply_file):
            point_clouds[method_dir] = ply_file
    
    return point_clouds


def create_file_index(root_dir: str, use_rectified: bool = True) -> Dict[int, Dict[str, List[str]]]:
    """Create comprehensive index of all DTU files organized by scan ID.
    
    Args:
        root_dir: Root directory of DTU dataset
        use_rectified: Whether to index rectified images
        
    Returns:
        index: Nested dictionary with structure:
               {scan_id: {"images": [...], "calibrations": [...], "point_clouds": {...}}}
    """
    index = {}
    
    # Get available scans
    available_scans = get_available_scans(root_dir, use_rectified)
    
    for scan_id in available_scans:
        try:
            paths = validate_dtu_structure(root_dir, scan_id, use_rectified)
            
            # Find image and calibration files
            images, calibs, view_ids = find_matching_files(paths['images'], paths['calibrations'])
            
            # Find point clouds if available
            point_clouds = {}
            if 'points' in paths:
                point_clouds = get_point_cloud_files(paths['points'], scan_id)
            
            index[scan_id] = {
                "images": images,
                "calibrations": calibs, 
                "view_ids": view_ids,
                "point_clouds": point_clouds,
                "lighting_conditions": get_lighting_conditions(paths['images'])
            }
            
        except (FileNotFoundError, RuntimeError):
            # Skip scans with missing or invalid data
            continue
    
    return index