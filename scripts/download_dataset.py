#!/usr/bin/env python3
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#     python3 scripts/download_dataset.py ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     python3 scripts/download_dataset.py WaterDrop /tmp/

import argparse
import os
import sys
import urllib.request
import urllib.error

def download_file(url, output_path):
    """Downloads a file from a URL to a local path."""
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Done.")
    except urllib.error.URLError as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        # We don't exit here because we might want to try other files or just report the error
        # but the bash script has `set -e`, so it exits on error.
        # I should exit on error to match behavior.
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download dataset for GNN Physics Simulator.")
    parser.add_argument("dataset_name", help="Name of the dataset to download (e.g., WaterDrop)")
    parser.add_argument("--output_dir", help="Directory where the dataset will be saved (default: project_root/data/)")
    
    args = parser.parse_args()
    
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Get the directory containing the script (scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the project root (parent of scripts/)
        project_root = os.path.dirname(script_dir)
        base_output_dir = os.path.join(project_root, 'data')

    dataset_name = args.dataset_name
    output_dir = os.path.join(base_output_dir, dataset_name)
    
    base_url = f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/"
    
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}", file=sys.stderr)
        sys.exit(1)
    
    files_to_download = [
        "metadata.json",
        "train.tfrecord",
        "valid.tfrecord",
        "test.tfrecord"
    ]
    
    for filename in files_to_download:
        file_url = f"{base_url}{filename}"
        output_path = os.path.join(output_dir, filename)
        download_file(file_url, output_path)

if __name__ == "__main__":
    main()
