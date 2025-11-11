"""
Data Extraction Script

This script extracts the data files from tar.gz archives.
Run this before running the baselines.

Usage:
    python extract_data.py
"""

import os
import tarfile
import zipfile
import sys

def extract_data():
    """Extract all data files from tar.gz archives."""
    
    # Creating data_extracted directory
    data_dir = 'data_extracted'
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created/verified directory: {data_dir}")
    
    # Files to extract
    files_to_extract = [
        ('training_top_100000_data.tar.gz', 'Training data'),
        ('testing_benchmarks.tar.gz', 'Test benchmarks'),
        ('testing_corpus.tar.gz', 'Test corpus'),
    ]
    
    # Extracting tar.gz files
    for tar_file, description in files_to_extract:
        if os.path.exists(tar_file):
            print(f"\nExtracting {description} from {tar_file}...")
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(data_dir)
                print(f"✓ Successfully extracted {tar_file}")
            except Exception as e:
                print(f"✗ Error extracting {tar_file}: {e}")
        else:
            print(f"⚠ {tar_file} not found, skipping...")
    
    # Extracting nested zip files
    print("\nExtracting nested zip files...")
    
    # Extracting benchmarks.zip
    benchmarks_zip = os.path.join(data_dir, 'benchmarks.zip')
    if os.path.exists(benchmarks_zip):
        benchmarks_dir = os.path.join(data_dir, 'benchmarks')
        os.makedirs(benchmarks_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(benchmarks_zip, 'r') as zip_ref:
                zip_ref.extractall(benchmarks_dir)
            print(f"✓ Successfully extracted benchmarks.zip")
        except Exception as e:
            print(f"✗ Error extracting benchmarks.zip: {e}")
    else:
        print("⚠ benchmarks.zip not found, skipping...")
    
    # Extracting corpus.zip
    corpus_zip = os.path.join(data_dir, 'corpus.zip')
    if os.path.exists(corpus_zip):
        corpus_dir = os.path.join(data_dir, 'corpus')
        os.makedirs(corpus_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(corpus_zip, 'r') as zip_ref:
                zip_ref.extractall(corpus_dir)
            print(f"✓ Successfully extracted corpus.zip")
        except Exception as e:
            print(f"✗ Error extracting corpus.zip: {e}")
    else:
        print("⚠ corpus.zip not found, skipping...")
    
    # Verifying extraction
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    
    # Checking for key files
    key_files = [
        os.path.join(data_dir, 'top_100000_data.csv'),
        os.path.join(data_dir, 'benchmarks', 'contractnli.json'),
        os.path.join(data_dir, 'corpus', 'contractnli'),
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                file_count = len([f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))])
                print(f"✓ {file_path} (directory with {file_count} files)")
            else:
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"✓ {file_path} ({size:.2f} MB)")
        else:
            print(f"✗ {file_path} NOT FOUND")
    
    # Listing directory structure
    print("\n" + "="*60)
    print("Directory structure:")
    print("="*60)
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        # Showing files (limiting to 5 per directory)
        for file in files[:5]:
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')
    
    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)
    print(f"\nData is now in: {os.path.abspath(data_dir)}")
    print("\nYou can now run:")
    print("  python simple-baseline.py --corpus data_extracted/corpus --benchmark data_extracted/benchmarks/contractnli.json --output output/tfidf.json")
    print("  python strong-baseline.py --corpus data_extracted/corpus --benchmark data_extracted/benchmarks/contractnli.json --output output/bm25.json")

if __name__ == '__main__':
    print("="*60)
    print("Data Extraction Script")
    print("="*60)
    print("\nThis script will extract data from tar.gz files.")
    print("Make sure you're in the project directory with the tar.gz files.")
    print("\nStarting extraction...")
    print("="*60)
    
    extract_data()
