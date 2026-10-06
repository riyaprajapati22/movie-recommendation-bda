"""Download and unzip MovieLens (small = 100k ratings, quick; use --size 25m for 'big' data)."""
import argparse, urllib.request, zipfile, os

URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "25m":   "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

p = argparse.ArgumentParser()
p.add_argument("--size", choices=URLS, default="small")
args = p.parse_args()

os.makedirs("data", exist_ok=True)
zip_path = f"data/ml-{args.size}.zip"
print("Downloading", URLS[args.size])
urllib.request.urlretrieve(URLS[args.size], zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall("data")
print("Done. Files are in ./data")
