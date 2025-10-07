""" This script is use to copy the evaluation results under the scratch folder to the results_backup folder.
Only the json file with evaluation results will be copied, other files like checkpoints, logs, etc. will not be copied.
"""
import shutil
import argparse
from pathlib import Path

def backup_eval_res(root_dir, target_filename, destination_dir, overwrite=False):
    # Go through all directories and find matching files
    all_eval_files = Path(root_dir).glob("*/evaluation/{}".format(target_filename))
    for eval_file in all_eval_files:
        src_file = eval_file
        dst_file = Path(destination_dir) / src_file.relative_to(root_dir)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        elif overwrite:
            shutil.copy2(src_file, dst_file)
            print(f"Overwritten: {src_file} -> {dst_file}")
        elif dst_file.exists():
            print(f"Skipped: {src_file} -> {dst_file} (already exists)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="scratch")
    parser.add_argument("--target-filename", type=str, default="final_evaluation_scores.json")
    parser.add_argument("--destination-dir", type=str, default="results_backup")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing files")
    args = parser.parse_args()
    
    print(f"Copying evaluation results from the experiment folders under {args.root_dir} to {args.destination_dir}")
    print("The directory structure will be preserved.")
    
    backup_eval_res(args.root_dir, args.target_filename, args.destination_dir, args.overwrite)
