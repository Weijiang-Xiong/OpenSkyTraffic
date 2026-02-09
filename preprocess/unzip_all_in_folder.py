"""Unzip all zip files in a folder into individual subfolders.

Usage:
    python preprocess/unzip_all_in_folder.py /path/to/folder
"""

import argparse
from pathlib import Path
from zipfile import ZipFile


def unzip_all_in_folder(folder: Path) -> None:
    zip_files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    )

    if not zip_files:
        print(f"No zip files found in {folder}")
        return

    for zip_path in zip_files:
        output_dir = folder / zip_path.stem
        output_dir.mkdir(exist_ok=True)

        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        print(f"Extracted {zip_path.name} -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unzip all zip files in a folder to individual subfolders."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing zip files (default: current directory).",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser()
    unzip_all_in_folder(folder)


if __name__ == "__main__":
    main()
