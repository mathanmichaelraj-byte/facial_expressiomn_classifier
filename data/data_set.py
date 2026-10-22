from pathlib import Path
import shutil

import kagglehub

DATASET = "msambare/fer2013"
SPLITS = ("train", "test")


def find_split(download_root, name):
    candidate = download_root / name
    if candidate.is_dir():
        return candidate
    matches = [path for path in download_root.glob(f"*/{name}") if path.is_dir()]
    return matches[0] if matches else None


def main():
    data_dir = Path(__file__).resolve().parent
    missing = [split for split in SPLITS if not (data_dir / split).is_dir()]

    if not missing:
        print(f"Dataset already present in {data_dir}")
        return

    print(f"Downloading {DATASET}...")
    download_root = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to: {download_root}")

    for split in missing:
        source = find_split(download_root, split)
        if source is None:
            raise FileNotFoundError(
                f"Could not find a '{split}' folder inside {download_root}"
            )
        destination = data_dir / split
        print(f"Copying {source} -> {destination}")
        shutil.copytree(source, destination)

    print("Dataset ready. Run 'python train.py' from the project root.")


if __name__ == "__main__":
    main()
