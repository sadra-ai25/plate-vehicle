#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import requests
import sys
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def send_plate_recognition_request(image_path: str, api_url: str = "http://127.0.0.1:3002/plate", side: str = None):
    """
    Send request to plate recognition API
    
    Args:
        image_path: Path to image file
        api_url: API address (default: http://127.0.0.1:3001/plate)
        side: Optional - 'R' for rightmost vehicle or 'L' for leftmost
    """
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File {image_path} not found!")
        return None
    
    # Convert image to base64
    print(f"Reading image: {image_path}")
    base64_image = image_to_base64(image_path)
    print(f"Image converted to base64 (length: {len(base64_image)} characters)")
    
    # Prepare JSON payload
    payload = {
        "image": base64_image
    }
    
    # Add side parameter if specified
    if side:
        payload["side"] = side
        print(f"{'Rightmost' if side == 'R' else 'Leftmost'} vehicle detection enabled")
    
    # Send request
    print(f"Sending request to {api_url}...")
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # 30 seconds timeout
        )
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Response received:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection error: Cannot connect to {api_url}")
        print("Please make sure the server is running")
        return None
    except requests.exceptions.Timeout:
        print("\n❌ Timeout error: No response from server")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return None


if __name__ == "__main__":
    # Command line usage example
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"python {sys.argv[0]} <image_path> [side]")
        print(f"Example: python {sys.argv[0]} ./car.jpg")
        print(f"Example with side: python {sys.argv[0]} ./car.jpg R")
        print("\nside can be:")
        print("  R = Rightmost vehicle")
        print("  L = Leftmost vehicle")
        sys.exit(1)
    
    image_path = sys.argv[1]
    side = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate side parameter
    if side and side not in ['R', 'L']:
        print("Warning: side must be 'R' (right) or 'L' (left)")
        side = None
    
    # Send request
    result = send_plate_recognition_request(image_path, side=side)
    
    # Save final output for use in other scripts
    if result:
        # Save to file (optional)
        output_file = f"result_{Path(image_path).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Result saved to file {output_file}")