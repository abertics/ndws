import ee
import os
import requests
import time

def main():
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        print("Error initializing Earth Engine. Have you authenticated? Error:", e)
        return

    # Get user input
    asset_id = input("Enter the asset ID (e.g., users/username/asset): ").strip()
    local_dir = input("Enter local directory to save (default: current directory): ").strip()
    
    # Use current directory if none specified
    if not local_dir:
        local_dir = os.getcwd()
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Get the asset
        image = ee.Image(asset_id)
        
        # Get download URL
        url = image.getDownloadURL({
            'scale': 30,  # Adjust resolution as needed
            'format': 'GeoTIFF',
            'region': image.geometry().bounds(),
            'maxPixels': 1e13
        })
        
        print(f"\nStarting download of {asset_id}...")
        
        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return
            
        # Define output filename
        filename = os.path.join(local_dir, f"{asset_id.split('/')[-1]}.tif")
        
        # Save the file with progress indicator
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                
                # Calculate progress
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rDownload progress: {progress:.1f}%", end='')
        
        print(f"\nDownload complete! File saved to: {filename}")
            
    except Exception as e:
        print(f"\nError during download: {e}")

if __name__ == "__main__":
    main()