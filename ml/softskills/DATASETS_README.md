# Soft Skills Datasets

This directory contains datasets for training and evaluating the soft skills analysis models.

## Directory Structure

```
datasets/
├── fluency/          # Speech fluency datasets (PodcastFillers, DisfluencySpeech)
├── gaze/             # Eye tracking datasets
├── gestures/         # Hand gesture datasets (IPN Hand)
├── posture/          # Body posture datasets (MPII Pose)
└── interview/        # Interview-specific datasets
```

## Priority Datasets

### P0 - Must Have

| Dataset | Directory | Size | Source |
|---------|-----------|------|--------|
| PodcastFillers | `fluency/podcast-fillers/` | ~500MB | [Adobe Research](https://github.com/adobe-research/podcast-fillers) |
| IPN Hand | `gestures/ipn-hand/` | ~2GB | [GitHub](https://gibranbenitez.github.io/IPN_Hand) |

### P1 - Important

| Dataset | Directory | Size | Source |
|---------|-----------|------|--------|
| Interview Candidate | `interview/kaggle-confidence/` | ~200MB | Kaggle |
| MPII Human Pose | `posture/mpii/` | ~12GB | [MPI](http://human-pose.mpi-inf.mpg.de/) |

### P2 - Nice to Have

| Dataset | Directory | Size | Source |
|---------|-----------|------|--------|
| DisfluencySpeech | `fluency/disfluency-speech/` | ~5GB | HuggingFace |
| SEP-28k | `fluency/sep-28k/` | ~3GB | Kaggle |

## Download Instructions

Run the download script:
```bash
python scripts/download_softskills_datasets.py --priority P0
```

Or manually:
```bash
# PodcastFillers
git clone https://github.com/adobe-research/podcast-fillers.git ml/softskills/datasets/fluency/podcast-fillers

# IPN Hand (large)
# Download from: https://gibranbenitez.github.io/IPN_Hand
```

## Licenses

- **PodcastFillers**: Adobe Research License (research use only)
- **IPN Hand**: Creative Commons (CC BY-NC 4.0)
- **MPII Human Pose**: Research use only
- **Interview Candidate**: Kaggle Open Database License

> **Note:** These datasets are for research and educational purposes only.
