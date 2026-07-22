import os
import requests
from ultralytics import YOLO

# 1. Ensure the 'models' directory exists safely
os.makedirs("models", exist_ok=True)

print("Downloading YOLO26 Nano Pose weights natively...")
url = "https://github.com"
output_path = "yolo26s-pose.pt"

# 2. Download via stream to avoid unpickling / HTML page errors
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Success! Model securely saved to: {output_path}")
else:
    print(f"Download failed. Status code: {response.status_code}")
