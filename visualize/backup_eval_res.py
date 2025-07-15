
import shutil
import argparse
from pathlib import Path

def backup_eval_res(root_dir, target_filename, destination_dir):
    # Go through all directories and find matching files
    all_eval_files = Path(root_dir).glob("*/evaluation/{}".format(target_filename))
    for eval_file in all_eval_files:
        src_file = eval_file
        dst_file = Path(destination_dir) / src_file.relative_to(root_dir)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="scratch")
    parser.add_argument("--target_filename", type=str, default="final_evaluation_scores.json")
    parser.add_argument("--destination_dir", type=str, default="results_backup")
    args = parser.parse_args()
    
    print(f"Copying evaluation results from the experiment folders under {args.root_dir} to {args.destination_dir}")
    print("The directory structure will be preserved.")
    
    backup_eval_res(args.root_dir, args.target_filename, args.destination_dir)
