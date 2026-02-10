"""Unzip all zip files in a folder into individual subfolders.

Usage:
    python preprocess/unzip_all_in_folder.py /path/to/folder
"""

import argparse
from pathlib import Path
from zipfile import ZipFile


def collapse_duplicate_folder_name(output_dir: Path) -> int:
    """Flatten output_dir/name/name/... to output_dir when applicable.

    Returns the number of duplicate levels collapsed.
    """
    collapsed_levels = 0

    while True:
        entries = list(output_dir.iterdir())
        duplicate_dir = output_dir / output_dir.name

        if len(entries) != 1 or entries[0] != duplicate_dir or not duplicate_dir.is_dir():
            break

        for child in list(duplicate_dir.iterdir()):
            child.rename(output_dir / child.name)

        duplicate_dir.rmdir()
        collapsed_levels += 1

    return collapsed_levels


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

        collapsed_levels = collapse_duplicate_folder_name(output_dir)

        print(f"Extracted {zip_path.name} -> {output_dir}")
        if collapsed_levels:
            print(f"Collapsed duplicated folder nesting x{collapsed_levels} in {output_dir}")


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
