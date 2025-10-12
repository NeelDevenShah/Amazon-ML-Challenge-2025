import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Read the training data
train_df = pd.read_csv('student_resource/dataset/train.csv')

# Create directories for saving images
os.makedirs('images/train', exist_ok=True)

# Create a session with retry strategy
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Function to download a single image
def download_image(row, session):
    image_url = row['image_link']
    sample_id = row['sample_id']
    
    try:
        image_response = session.get(image_url, timeout=10)
        if image_response.status_code == 200:
            with open(f'images/train/{sample_id}.jpg', 'wb') as f:
                f.write(image_response.content)
            return (sample_id, True, None)
        else:
            return (sample_id, False, f"Status code: {image_response.status_code}")
    except Exception as e:
        return (sample_id, False, str(e))

# Download images concurrently
fails = []
session = create_session()

# Use ThreadPoolExecutor for concurrent downloads (adjust max_workers based on your needs)
max_workers = 20  # You can increase this for faster downloads (e.g., 50)

print(f"Starting download of {len(train_df)} images with {max_workers} workers...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all download tasks
    futures = {executor.submit(download_image, row, session): row for _, row in train_df.iterrows()}
    
    # Process completed downloads with progress bar
    with tqdm(total=len(train_df), desc="Downloading images") as pbar:
        for future in as_completed(futures):
            sample_id, success, error = future.result()
            if not success:
                fails.append({'sample_id': sample_id, 'error': error})
                print(f"\nFailed to download image for sample_id: {sample_id} - Error: {error}")
            pbar.update(1)

print(f"\nDownload complete! Successfully downloaded {len(train_df) - len(fails)} images.")
if fails:
    print(f"Failed to download {len(fails)} images.")
    # Save failed downloads to a CSV for review
    pd.DataFrame(fails).to_csv('failed_downloads.csv', index=False)
    print("Failed downloads saved to 'failed_downloads.csv'")