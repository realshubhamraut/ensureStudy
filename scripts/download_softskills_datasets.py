#!/usr/bin/env python3
"""
Soft Skills Datasets Download Script

Downloads datasets needed for the soft skills evaluation module.
Priority levels:
- P0: Essential datasets (download first)
- P1: Important datasets
- P2: Nice to have

Usage:
    python scripts/download_softskills_datasets.py --priority P0
    python scripts/download_softskills_datasets.py --all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "ml" / "softskills" / "datasets"


# Dataset configurations
DATASETS = {
    # Priority 0 - Must Have
    "podcast-fillers-utils": {
        "priority": "P0",
        "type": "git",
        "url": "https://github.com/gzhu06/PodcastFillers_Utils.git",
        "destination": DATASETS_DIR / "fluency" / "podcast-fillers-utils",
        "description": "PodcastFillers preprocessing utils and training code",
        "size": "~50MB"
    },
    "podcast-fillers-hf": {
        "priority": "P0",
        "type": "huggingface",
        "repo": "ylacombe/podcast_fillers_by_license",
        "destination": DATASETS_DIR / "fluency" / "podcast-fillers-hf",
        "description": "PodcastFillers subset from HuggingFace",
        "size": "~500MB"
    },
    "hagrid-gestures": {
        "priority": "P0",
        "type": "git",
        "url": "https://github.com/hukenovs/hagrid.git",
        "destination": DATASETS_DIR / "gestures" / "hagrid",
        "description": "Hand Gesture Recognition Image Dataset (HaGRID)",
        "size": "~100MB (repo only, images separate)"
    },
    
    # Priority 1 - Important
    "interview-candidate": {
        "priority": "P1",
        "type": "kaggle",
        "dataset": "lauradonato/interview-candidate-dataset",
        "destination": DATASETS_DIR / "interview" / "kaggle-confidence",
        "description": "Images of confident/not confident candidates",
        "size": "~200MB"
    },
    "mediapipe-gesture-recognizer": {
        "priority": "P1",
        "type": "manual",
        "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        "destination": DATASETS_DIR / "gestures" / "mediapipe-models",
        "description": "Pre-trained MediaPipe gesture recognition model",
        "size": "~5MB"
    },
    
    # Priority 2 - Nice to Have
    "disfluency-speech": {
        "priority": "P2",
        "type": "huggingface",
        "repo": "speech-colab/disfluency-speech",
        "destination": DATASETS_DIR / "fluency" / "disfluency-speech",
        "description": "10 hours of labeled disfluency audio",
        "size": "~5GB"
    },
    "common-voice-sample": {
        "priority": "P2",
        "type": "huggingface",
        "repo": "mozilla-foundation/common_voice_11_0",
        "destination": DATASETS_DIR / "fluency" / "common-voice",
        "description": "Mozilla Common Voice for baseline speech",
        "size": "~50GB (subset only)"
    }
}


def check_git():
    """Check if git is available"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_kaggle_cli():
    """Check if Kaggle CLI is installed and configured"""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_git_dataset(config):
    """Clone a git repository"""
    dest = config["destination"]
    url = config["url"]
    
    if dest.exists():
        print(f"  ℹ️  Already exists: {dest}")
        return True
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"  📥 Cloning from {url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True
        )
        print(f"  ✅ Downloaded to {dest}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to clone: {e}")
        return False


def download_kaggle_dataset(config):
    """Download a Kaggle dataset"""
    if not check_kaggle_cli():
        print("  ⚠️  Kaggle CLI not installed. Run: pip install kaggle")
        print(f"     Then configure: ~/.kaggle/kaggle.json")
        print(f"     Manual download: https://www.kaggle.com/datasets/{config['dataset']}")
        return False
    
    dest = config["destination"]
    dataset = config["dataset"]
    
    if dest.exists() and any(dest.iterdir()):
        print(f"  ℹ️  Already exists: {dest}")
        return True
    
    dest.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"  📥 Downloading {dataset}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
            check=True
        )
        print(f"  ✅ Downloaded to {dest}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to download: {e}")
        return False


def download_huggingface_dataset(config):
    """Download a HuggingFace dataset"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ⚠️  huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    dest = config["destination"]
    repo = config["repo"]
    
    if dest.exists() and any(dest.iterdir()):
        print(f"  ℹ️  Already exists: {dest}")
        return True
    
    dest.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"  📥 Downloading {repo}...")
        snapshot_download(repo_id=repo, local_dir=str(dest), repo_type="dataset")
        print(f"  ✅ Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download: {e}")
        return False


def show_manual_instructions(config):
    """Show instructions for manual downloads"""
    print(f"  📋 Manual download required:")
    print(f"     URL: {config['url']}")
    print(f"     Destination: {config['destination']}")
    if "instructions" in config:
        print(f"     Note: {config['instructions']}")
    return False


def download_dataset(name, config):
    """Download a single dataset based on its type"""
    print(f"\n📦 {name} ({config['priority']}) - {config['description']}")
    print(f"   Size: {config['size']}")
    
    dtype = config["type"]
    
    if dtype == "git":
        return download_git_dataset(config)
    elif dtype == "kaggle":
        return download_kaggle_dataset(config)
    elif dtype == "huggingface":
        return download_huggingface_dataset(config)
    elif dtype == "manual":
        return show_manual_instructions(config)
    else:
        print(f"  ❌ Unknown type: {dtype}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download soft skills datasets")
    parser.add_argument(
        "--priority",
        choices=["P0", "P1", "P2"],
        help="Download datasets of specific priority"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Download specific dataset by name"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📚 Available Datasets:\n")
        for name, config in DATASETS.items():
            print(f"  [{config['priority']}] {name}")
            print(f"      {config['description']}")
            print(f"      Size: {config['size']}, Type: {config['type']}")
            print()
        return
    
    # Ensure base directories exist
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter datasets based on arguments
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"❌ Unknown dataset: {args.dataset}")
            print(f"   Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        to_download = {args.dataset: DATASETS[args.dataset]}
    elif args.priority:
        to_download = {
            name: config
            for name, config in DATASETS.items()
            if config["priority"] == args.priority
        }
    elif args.all:
        to_download = DATASETS
    else:
        # Default to P0
        print("ℹ️  No priority specified, downloading P0 (essential) datasets...")
        to_download = {
            name: config
            for name, config in DATASETS.items()
            if config["priority"] == "P0"
        }
    
    if not to_download:
        print("No datasets to download.")
        return
    
    print(f"\n🚀 Downloading {len(to_download)} dataset(s)...\n")
    
    results = {}
    for name, config in to_download.items():
        results[name] = download_dataset(name, config)
    
    # Summary
    print("\n" + "="*50)
    print("📊 Download Summary:")
    print("="*50)
    
    success = [n for n, r in results.items() if r]
    failed = [n for n, r in results.items() if not r]
    
    if success:
        print(f"✅ Success: {', '.join(success)}")
    if failed:
        print(f"❌ Failed/Manual: {', '.join(failed)}")


if __name__ == "__main__":
    main()
