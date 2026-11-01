import requests
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

url = (
    "https://archive.ics.uci.edu/"
    "static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

zip_path = DATA_DIR / "household_data.zip"

print("Downloading electricity dataset...")

response = requests.get(url, timeout=120)
response.raise_for_status()

zip_path.write_bytes(response.content)

print("Extracting dataset...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("Dataset downloaded successfully!")
print("Location:", DATA_DIR.resolve())