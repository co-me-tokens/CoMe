"""Path management utilities for DTU dataset.

This module provides a clean, simple interface for managing DTU dataset paths
without the over-engineering of the original fragmented implementation.
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DTUPaths:
    """Simple path manager for DTU dataset structure.
    
    This replaces the fragmented path management from the original code
    with a clean, straightforward implementation.
    """
    
    root: str
    scan_id: int
    use_rectified: bool = True
    calibration_set: str = "cal18"
    
    def __post_init__(self):
        """Validate that root directory exists."""
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"DTU root directory not found: {self.root}")
    
    @property 
    def image_directory(self) -> str:
        """Get path to image directory for this scan."""
        subdir = "Rectified" if self.use_rectified else "Cleaned"
        return os.path.join(self.root, subdir, f"scan{self.scan_id}")
    
    @property
    def calibration_directory(self) -> str:
        """Get path to calibration directory."""
        return os.path.join(self.root, "Calibration", self.calibration_set)
    
    @property  
    def points_directory(self) -> str:
        """Get base path to points directory."""
        return os.path.join(self.root, "Points")
    
    def points_method_directory(self, method: str) -> str:
        """Get path to points directory for specific reconstruction method.
        
        Args:
            method: Reconstruction method name (e.g., "stl", "mvsnet")
            
        Returns:
            Path to method-specific points directory
        """
        return os.path.join(self.points_directory, method)
    
    def point_cloud_file(self, method: str) -> str:
        """Get path to point cloud PLY file for this scan.
        
        Args:
            method: Reconstruction method name
            
        Returns:
            Path to PLY file following DTU naming convention
        """
        filename = f"{method}{self.scan_id:03d}_total.ply"
        return os.path.join(self.points_method_directory(method), filename)
    
    def validate_structure(self) -> dict:
        """Validate DTU directory structure and return status.
        
        Returns:
            Dictionary with validation results for each component
        """
        results = {}
        
        # Check image directory
        results['images'] = os.path.isdir(self.image_directory)
        
        # Check calibration directory  
        results['calibrations'] = os.path.isdir(self.calibration_directory)
        
        # Check points directory (optional)
        results['points'] = os.path.isdir(self.points_directory)
        
        # Check for available point cloud methods
        available_methods = []
        if results['points']:
            try:
                for item in os.listdir(self.points_directory):
                    method_dir = os.path.join(self.points_directory, item)
                    if os.path.isdir(method_dir):
                        ply_file = self.point_cloud_file(item)
                        if os.path.exists(ply_file):
                            available_methods.append(item)
            except OSError:
                pass
        
        results['point_cloud_methods'] = available_methods
        
        return results
    
    def get_summary(self) -> str:
        """Get human-readable summary of DTU paths configuration."""
        validation = self.validate_structure()
        
        summary = [f"DTU Paths Summary for Scan {self.scan_id}:"]
        summary.append(f"  Root: {self.root}")
        summary.append(f"  Image type: {'Rectified' if self.use_rectified else 'Cleaned'}")
        summary.append(f"  Calibration set: {self.calibration_set}")
        summary.append("")
        summary.append("Directory validation:")
        summary.append(f"  Images: {'✓' if validation['images'] else '✗'} {self.image_directory}")
        summary.append(f"  Calibrations: {'✓' if validation['calibrations'] else '✗'} {self.calibration_directory}")
        summary.append(f"  Points: {'✓' if validation['points'] else '✗'} {self.points_directory}")
        
        if validation['point_cloud_methods']:
            summary.append("  Available point cloud methods:")
            for method in validation['point_cloud_methods']:
                summary.append(f"    - {method}: {self.point_cloud_file(method)}")
        else:
            summary.append("  No point cloud files found")
        
        return "\n".join(summary)


def get_all_dtu_scans(root: str, use_rectified: bool = True) -> List[int]:
    """Get list of all available DTU scan IDs.
    
    Args:
        root: DTU dataset root directory
        use_rectified: Whether to look for rectified images
        
    Returns:
        List of available scan IDs
    """
    subdir = "Rectified" if use_rectified else "Cleaned"
    images_base = os.path.join(root, subdir)
    
    if not os.path.exists(images_base):
        return []
    
    scan_ids = []
    for item in os.listdir(images_base):
        if item.startswith("scan") and os.path.isdir(os.path.join(images_base, item)):
            try:
                scan_id = int(item[4:])  # Remove "scan" prefix
                scan_ids.append(scan_id)
            except ValueError:
                continue
    
    return sorted(scan_ids)


def validate_dtu_installation(root: str) -> dict:
    """Validate overall DTU dataset installation.
    
    Args:
        root: DTU dataset root directory
        
    Returns:
        Dictionary with validation results and statistics
    """
    if not os.path.exists(root):
        return {"valid": False, "error": f"Root directory not found: {root}"}
    
    results = {
        "valid": True,
        "root": root,
        "rectified_scans": get_all_dtu_scans(root, use_rectified=True),
        "cleaned_scans": get_all_dtu_scans(root, use_rectified=False),
        "calibration_sets": [],
        "point_cloud_methods": []
    }
    
    # Check calibration sets
    calib_dir = os.path.join(root, "Calibration")
    if os.path.exists(calib_dir):
        results["calibration_sets"] = [
            item for item in os.listdir(calib_dir) 
            if os.path.isdir(os.path.join(calib_dir, item))
        ]
    
    # Check point cloud methods
    points_dir = os.path.join(root, "Points")
    if os.path.exists(points_dir):
        results["point_cloud_methods"] = [
            item for item in os.listdir(points_dir)
            if os.path.isdir(os.path.join(points_dir, item))
        ]
    
    # Summary statistics
    results["total_rectified_scans"] = len(results["rectified_scans"])
    results["total_cleaned_scans"] = len(results["cleaned_scans"])
    results["total_calibration_sets"] = len(results["calibration_sets"])
    results["total_point_cloud_methods"] = len(results["point_cloud_methods"])
    
    return results


def create_dtu_paths(root: str, scan_id: int, use_rectified: bool = True, 
                    calibration_set: str = "cal18") -> DTUPaths:
    """Factory function to create DTUPaths with validation.
    
    Args:
        root: DTU dataset root directory
        scan_id: Scan ID (1-128 typically)
        use_rectified: Whether to use rectified images  
        calibration_set: Calibration set name
        
    Returns:
        DTUPaths instance
        
    Raises:
        FileNotFoundError: If root directory doesn't exist
        ValueError: If scan_id is invalid
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"DTU root directory not found: {root}")
    
    available_scans = get_all_dtu_scans(root, use_rectified)
    if scan_id not in available_scans:
        image_type = "rectified" if use_rectified else "cleaned"
        raise ValueError(
            f"Scan {scan_id} not found in {image_type} images. "
            f"Available scans: {available_scans}"
        )
    
    return DTUPaths(
        root=root,
        scan_id=scan_id, 
        use_rectified=use_rectified,
        calibration_set=calibration_set
    )


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        dtu_root = sys.argv[1]
    else:
        dtu_root = "/Dataset/DTU_MVS"  # Default path
    
    print("DTU Dataset Validation")
    print("=" * 50)
    
    # Validate installation
    validation = validate_dtu_installation(dtu_root)
    
    if not validation["valid"]:
        print(f"❌ {validation['error']}")
        sys.exit(1)
    
    print(f"✅ DTU dataset found at: {dtu_root}")
    print(f"📁 Rectified scans: {validation['total_rectified_scans']}")  
    print(f"📁 Cleaned scans: {validation['total_cleaned_scans']}")
    print(f"📋 Calibration sets: {validation['total_calibration_sets']}")
    print(f"☁️ Point cloud methods: {validation['total_point_cloud_methods']}")
    
    if validation["rectified_scans"]:
        print("\nTesting scan 1 paths...")
        try:
            paths = create_dtu_paths(dtu_root, scan_id=1)
            print(paths.get_summary())
        except Exception as e:
            print(f"❌ Error creating paths: {e}")
    else:
        print("⚠️ No rectified scans found for testing")