#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import requests
import sys
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def send_single_image(image_path: str, api_url: str, side: str = None) -> dict:
    """
    Send a single image to the API and return the result
    
    Args:
        image_path: Path to image file
        api_url: API endpoint URL
        side: Optional - 'R' for rightmost or 'L' for leftmost vehicle
    """
    try:
        base64_image = image_to_base64(image_path)
        
        payload = {"image": base64_image}
        if side:
            payload["side"] = side
        
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "message": response.text}
            
    except Exception as e:
        return {"error": "Exception", "message": str(e)}


def process_folder(folder_path: str, api_url: str = "http://127.0.0.1:3001/plate", 
                   side: str = None, max_workers: int = 1):
    """
    Process all images in a folder and save JSON results
    
    Args:
        folder_path: Path to folder containing images
        api_url: API endpoint URL
        side: Optional side parameter (R/L)
        max_workers: Number of parallel threads (default 1 for sequential processing)
    """
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder {folder_path} does not exist or is not a directory")
        return
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all image files
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"API URL: {api_url}")
    if side:
        print(f"Side filter: {side}")
    print("-" * 50)
    
    def process_one_image(image_file: Path):
        """Process a single image and save result"""
        print(f"Processing: {image_file.name}...")
        
        result = send_single_image(str(image_file), api_url, side)
        
        # Save JSON with same name as image
        json_filename = image_file.stem + ".json"
        json_path = folder / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        status = "✓" if "error" not in result else "✗"
        print(f"  {status} Saved: {json_filename}")
        
        return image_file.name, "success" if "error" not in result else "failed"
    
    # Process images
    results = []
    if max_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_image, img): img for img in image_files}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        # Sequential processing
        for img_file in image_files:
            results.append(process_one_image(img_file))
    
    # Summary
    print("-" * 50)
    successful = sum(1 for _, status in results if status == "success")
    failed = len(results) - successful
    print(f"Completed: {successful} successful, {failed} failed")
    print(f"JSON files saved in: {folder_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"python {sys.argv[0]} <folder_path> [side] [api_url] [threads]")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} ./images")
        print(f"  python {sys.argv[0]} ./images R")
        print(f"  python {sys.argv[0]} ./images R http://192.168.1.100:3001/plate")
        print(f"  python {sys.argv[0]} ./images R http://127.0.0.1:3001/plate 4")
        print()
        print("Parameters:")
        print("  folder_path: Path to folder containing images")
        print("  side:        Optional - 'R' (rightmost) or 'L' (leftmost)")
        print("  api_url:     Optional - API endpoint (default: http://127.0.0.1:3001/plate)")
        print("  threads:     Optional - Number of parallel threads (default: 1)")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    side = sys.argv[2] if len(sys.argv) > 2 else None
    api_url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:3001/plate"
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    # Validate side
    if side and side not in ['R', 'L']:
        print(f"Warning: Invalid side '{side}', ignoring. Use 'R' or 'L'")
        side = None
    
    process_folder(folder_path, api_url, side, threads)