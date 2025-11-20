#!/usr/bin/env python3
"""
KITTI RGB Image Preprocessing Script

This script copies RGB images from KITTI_Raw dataset to KITTI_Depth dataset structure
to enable easier dataloader implementation. It matches the directory structure and
naming conventions of the depth dataset.

Usage:
    python preprocess.py
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def find_corresponding_rgb_images(kitti_raw_root, date, drive, camera, frame_ids):
    """
    Find corresponding RGB images in KITTI_Raw dataset.
    
    Args:
        kitti_raw_root: Path to KITTI_Raw dataset root
        date: Date string (e.g., '2011_09_28')
        drive: Drive string (e.g., '2011_09_28_drive_0167_sync')
        camera: Camera string (e.g., 'image_02')
        frame_ids: List of frame IDs to copy
    
    Returns:
        List of (source_path, frame_id) tuples
    """
    rgb_images = []
    rgb_dir = Path(kitti_raw_root) / date / drive / camera / "data"
    
    if not rgb_dir.exists():
        print(f"Warning: RGB directory not found: {rgb_dir}")
        return rgb_images
    
    for frame_id in frame_ids:
        rgb_file = rgb_dir / f"{frame_id:010d}.png"
        if rgb_file.exists():
            rgb_images.append((str(rgb_file), frame_id))
        else:
            print(f"Warning: RGB image not found: {rgb_file}")
    
    return rgb_images


def get_frame_ids_from_depth_dir(depth_dir):
    """
    Extract frame IDs from depth directory.
    
    Args:
        depth_dir: Path to depth directory
    
    Returns:
        List of frame IDs (integers)
    """
    frame_ids = []
    for depth_file in sorted(depth_dir.glob("*.png")):
        frame_id = int(depth_file.stem)
        frame_ids.append(frame_id)
    return frame_ids


def copy_rgb_images(kitti_raw_root, kitti_depth_root, output_root, split='train'):
    """
    Copy RGB images from KITTI_Raw to match KITTI_Depth structure.
    
    Args:
        kitti_raw_root: Path to KITTI_Raw dataset root
        kitti_depth_root: Path to KITTI_Depth dataset root  
        output_root: Path to output directory for RGB images
        split: 'train' or 'val'
    """
    split_dir = Path(kitti_depth_root) / split
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return
    
    output_split_dir = Path(output_root) / split
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all drive directories in the split
    drive_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    
    for drive_dir in tqdm(drive_dirs, desc=f"Processing {split} drives"):
        drive_name = drive_dir.name
        date = drive_name.split('_drive_')[0]
        
        # Look for depth directories
        depth_root = drive_dir / "proj_depth" / "groundtruth"
        if not depth_root.exists():
            print(f"Warning: Depth root not found: {depth_root}")
            continue
        
        # Process each camera
        camera_dirs = [d for d in depth_root.iterdir() if d.is_dir() and d.name.startswith('image_')]
        
        for camera_dir in camera_dirs:
            camera = camera_dir.name
            
            # Get frame IDs from depth files
            frame_ids = get_frame_ids_from_depth_dir(camera_dir)
            if not frame_ids:
                continue
            
            # Find corresponding RGB images
            rgb_images = find_corresponding_rgb_images(
                kitti_raw_root, date, drive_name, camera, frame_ids
            )
            
            if not rgb_images:
                print(f"Warning: No RGB images found for {drive_name}/{camera}")
                continue
            
            # Create output directory
            output_camera_dir = output_split_dir / drive_name / "image" / camera
            output_camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy RGB images
            for rgb_path, frame_id in tqdm(rgb_images, desc=f"{drive_name}/{camera}", leave=False):
                output_file = output_camera_dir / f"{frame_id:010d}.png"
                if not output_file.exists():
                    shutil.copy2(rgb_path, output_file)


def main():
    parser = argparse.ArgumentParser(description="Preprocess KITTI RGB images")
    parser.add_argument(
        "--kitti-raw-root", 
        type=str, 
        default="{YOUR_PATH_TO}/KITTI_Raw/",
        help="Path to KITTI_Raw dataset root"
    )
    parser.add_argument(
        "--kitti-depth-root",
        type=str, 
        default="{YOUR_PATH_TO}/KITTI_Depth/",
        help="Path to KITTI_Depth dataset root"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="{YOUR_PATH_TO}/KITTI_Depth/",
        help="Path to output directory for RGB images"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process"
    )
    
    args = parser.parse_args()
    
    print("KITTI RGB Image Preprocessing")
    print(f"KITTI_Raw root: {args.kitti_raw_root}")
    print(f"KITTI_Depth root: {args.kitti_depth_root}")
    print(f"Output root: {args.output_root}")
    print(f"Processing splits: {args.splits}")
    
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        copy_rgb_images(
            args.kitti_raw_root,
            args.kitti_depth_root, 
            args.output_root,
            split
        )
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()