# Data Extraction Guide

## How to Extract Data Files

The `data_extracted` folder is created by extracting the tar.gz files in your project directory. Here are the exact commands:

### Using Terminal/Command Line

```bash
# Navigate to project directory
cd "/Users/johnharshith/Downloads/NLP Project"

# Create data_extracted directory
mkdir -p data_extracted

# Extract training data
tar -xzf training_top_100000_data.tar.gz -C data_extracted

# Extract test benchmarks
tar -xzf testing_benchmarks.tar.gz -C data_extracted

# Extract test corpus
tar -xzf testing_corpus.tar.gz -C data_extracted

# Unzip nested zip files (if they exist inside the tar.gz files)
cd data_extracted
unzip -q benchmarks.zip -d benchmarks 2>/dev/null || echo "Already extracted or not found"
unzip -q corpus.zip -d corpus 2>/dev/null || echo "Already extracted or not found"
cd ..

# Verify extraction
ls -la data_extracted/
```

### Using Python (in Colab or local)

```python
import os
import tarfile
import zipfile

# Create data_extracted directory
os.makedirs('data_extracted', exist_ok=True)

# Extract training data
if os.path.exists('training_top_100000_data.tar.gz'):
    with tarfile.open('training_top_100000_data.tar.gz', 'r:gz') as tar:
        tar.extractall('data_extracted')
    print("Training data extracted")

# Extract test benchmarks
if os.path.exists('testing_benchmarks.tar.gz'):
    with tarfile.open('testing_benchmarks.tar.gz', 'r:gz') as tar:
        tar.extractall('data_extracted')
    print("Benchmarks extracted")

# Extract test corpus
if os.path.exists('testing_corpus.tar.gz'):
    with tarfile.open('testing_corpus.tar.gz', 'r:gz') as tar:
        tar.extractall('data_extracted')
    print("Corpus extracted")

# Unzip nested zip files
data_dir = 'data_extracted'
if os.path.exists(os.path.join(data_dir, 'benchmarks.zip')):
    with zipfile.ZipFile(os.path.join(data_dir, 'benchmarks.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, 'benchmarks'))
    print("Benchmarks zip extracted")

if os.path.exists(os.path.join(data_dir, 'corpus.zip')):
    with zipfile.ZipFile(os.path.join(data_dir, 'corpus.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, 'corpus'))
    print("Corpus zip extracted")

# Verify
print("\nDirectory structure:")
for root, dirs, files in os.walk(data_extracted):
    level = root.replace(data_extracted, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')
```

## Expected Directory Structure

After extraction, you should have:

```
data_extracted/
├── top_100000_data.csv          
├── benchmarks/                   
│   ├── contractnli.json
│   ├── cuad.json
│   ├── maud.json
│   └── privacy_qa.json
└── corpus/                      
    ├── contractnli/
    │   ├── file1.txt
    │   ├── file2.txt
    │   └── ...
    ├── cuad/
    ├── maud/
    └── privacy_qa/
```

This creates the `data_extracted` folder with all the necessary data files organized in subdirectories.
