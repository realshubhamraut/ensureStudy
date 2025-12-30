---
dataset_info:
  features:
  - name: license
    dtype: string
  - name: audio
    dtype: audio
  - name: file_name
    dtype: string
  - name: episode_name
    dtype: string
  - name: original_split
    dtype: string
  - name: sampling_rate
    dtype: int64
  - name: audio_length
    dtype: float64
  splits:
  - name: CC_BY_3.0
    num_bytes: 4245155531.5879397
    num_examples: 100
  - name: CC_BY_SA_3.0
    num_bytes: 3168500218.8944726
    num_examples: 79
  - name: CC_BY_ND_3.0
    num_bytes: 961275962.5175879
    num_examples: 20
  download_size: 8343014658
  dataset_size: 8374931713.0
configs:
- config_name: default
  data_files:
  - split: CC_BY_3.0
    path: data/CC_BY_3.0-*
  - split: CC_BY_SA_3.0
    path: data/CC_BY_SA_3.0-*
  - split: CC_BY_ND_3.0
    path: data/CC_BY_ND_3.0-*
license: cc
---

# Some Podcasts

Podcasts are taken from the [PodcastFillers dataset](https://podcastfillers.github.io/). The PodcastFillers dataset consists of 199 full-length podcast episodes in English with manually annotated filler words and automatically generated transcripts. The podcast audio recordings, sourced from SoundCloud, are CC-licensed, gender-balanced, and total 145 hours of audio from over 350 speakers.

> [!TIP]
> This dataset doesn't upload the PodcastFillers annotations, which are under a non-commercial license. See [here](https://podcastfillers.github.io/license/) for more details.


## Length by license type

**CC_BY 3.0:**
Total length: 73.6 h. Mean length: 44.2 min

**CC_BY SA 3.0:**
Total length: 54.9 h. Mean length: 41.7 min

**CC_BY ND 3.0 :**
Total length: 16.7 h. Mean length: 50 min

## License

See [here](https://podcastfillers.github.io/license/) for more details. The licenses are also in the metadata.

## Citation Information

```
@inproceedings{Zhu:FillerWords:INTERSPEECH:22,
  title = {Filler Word Detection and Classification: A Dataset and Benchmark},
  booktitle = {23rd Annual Cong.~of the Int.~Speech Communication Association (INTERSPEECH)},
  address = {Incheon, Korea}, 
  month = {Sep.},
  url = {https://arxiv.org/abs/2203.15135},
  author = {Zhu, Ge and Caceres, Juan-Pablo and Salamon, Justin},
  year = {2022},
}
```

### Contributions

Thanks to [@ylacombe](https://github.com/ylacombe) for adding this dataset.