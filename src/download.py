import os
import shutil
import kagglehub


def main():
    cache_path = kagglehub.dataset_download(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    )

    os.makedirs("data/raw", exist_ok=True)

    downloaded_file = os.path.join(cache_path, "IMDB Dataset.csv")
    target_file = "data/raw/dataset.csv"

    shutil.copy2(downloaded_file, target_file)


if __name__ == "__main__":
    main()
