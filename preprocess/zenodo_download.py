""" A really simple script for downloading a record(dataset) from Zenodo. 
    Usage: 
    
    python preprocess/zenodo_download.py <record_id> -o <output_directory>

    After downloading, you can check the md5 of the files and compare with the checksums
    provided by Zenodo.

    md5sum <filename>
"""
import argparse
import os
import subprocess

import requests

API = "https://zenodo.org/api/records/{}"


def get_record_json(record_id: str) -> dict:
    r = requests.get(API.format(record_id), timeout=60)
    r.raise_for_status()
    return r.json()


def extract_files(rec: dict):
    """
    Returns list of (filename, url, size_bytes, checksum).
    Handles common Zenodo API shapes:
      - rec["files"] is a list of file dicts
      - rec["files"]["entries"] is a dict of entries
    """
    out = []

    files = rec.get("files")
    if isinstance(files, list):
        for f in files:
            name = f.get("key") or f.get("filename")
            links = f.get("links") or {}
            url = links.get("download") or links.get("content") or links.get("self")
            size = f.get("size")
            checksum = f.get("checksum")
            if name and url:
                out.append((name, url, size, checksum))
        return out

    if isinstance(files, dict) and isinstance(files.get("entries"), dict):
        for name, meta in files["entries"].items():
            links = meta.get("links") or {}
            url = links.get("content") or links.get("download") or links.get("self")
            fname = meta.get("key") or name
            size = meta.get("size")
            checksum = meta.get("checksum")
            if fname and url:
                out.append((fname, url, size, checksum))
        return out

    raise RuntimeError("Unrecognized record JSON structure; cannot find files.")


def download(url: str, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    subprocess.run(["wget", "-O", path, "-c", url], check=True)


def parse_md5(checksum: str):
    if not checksum:
        return None
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1].strip().lower()
    if ":" in checksum:
        return None
    return checksum.strip().lower()


def main():
    parser = argparse.ArgumentParser(
        description="Minimal Zenodo bulk downloader (requests only).",
        epilog="Example: python preprocess/zenodo_download.py 10491409 -o ./datasets",
    )
    parser.add_argument("record", help="Record id e.g., 10491409")
    parser.add_argument("-o", "--outdir", default="zenodo_downloads")
    args = parser.parse_args()

    rec = get_record_json(args.record)
    files = extract_files(rec)

    print(f"Record {args.record}: {len(files)} files")
    for i, (name, url, size, checksum) in enumerate(files, 1):
        out_path = os.path.join(args.outdir, name)
        size_str = f"{size} bytes" if size is not None else "size unknown"
        md5_str = parse_md5(checksum) or "unknown"
        print(f"[{i}/{len(files)}] {name} ({size_str}) md5={md5_str}")
        download(url, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
