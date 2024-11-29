import ee
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
    bucket = input("Enter the destination bucket name (without gs://): ").strip()
    folder = input("Enter the destination folder path (optional): ").strip()
    
    # Construct the file name prefix
    prefix = folder + "/" + asset_id.split('/')[-1] if folder else asset_id.split('/')[-1]
    
    try:
        # Create the export task
        task = ee.batch.Export.image.toCloudStorage(
            image=ee.Image(asset_id),
            description=f'export',
            bucket=bucket,
            fileNamePrefix=prefix,
            scale=30,
            maxPixels=1e13
        )
        
        # Start the task
        task.start()
        
        print(f"\nExport started!")
        print(f"Asset: {asset_id}")
        print(f"Destination: gs://{bucket}/{prefix}")
        
        # Monitor the task for up to 1 minute
        print("\nMonitoring task status for 60 seconds...")
        for _ in range(2):  # Check status every 5 seconds for 1 minute
            status = task.status()['state']
            print(f"Status: {status}")
            if status == 'COMPLETED':
                print("Export completed successfully!")
                break
            elif status in ['FAILED', 'CANCELLED']:
                print(f"Export {status.lower()}!")
                break
            time.sleep(5)
        
        print("\nYou can continue monitoring the task in the Earth Engine Code Editor:")
        print("https://code.earthengine.google.com/tasks")
            
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == "__main__":
    main()