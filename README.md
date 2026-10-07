# SF-CLIP

Code for **SF-CLIP: Semantic Stage Alignment and Multimodal Fusion for Few-Shot Action Recognition**.

SF-CLIP aligns ordered semantic stage descriptions with video frames and matches query videos to support videos through ordered temporal alignment. It adapts the CLIP visual encoder while keeping text encoding frozen, and geometrically fuses semantic and visual predictions at inference.

## Performance

**Results reported in the paper.** All results use CLIP ViT-B/16 under 5-way settings and report mean top-1 accuracy over 10,000 sampled episodes.

| Dataset | 1-shot (%) | 5-shot (%) |
| --- | ---: | ---: |
| HMDB51 | 82.7 | 88.4 |
| UCF101 | 97.4 | 99.2 |
| Kinetics-100 | 93.2 | 95.5 |

The bundled configurations are usage examples. Legacy RN50 configurations do not correspond to the ViT-B/16 results above.

## Environment

Use CUDA-enabled PyTorch with a matching torchvision build. Multi-GPU examples use the NCCL backend on Linux. [environment.yaml](environment.yaml) provides historical dependency versions; the current code also requires:

```bash
python -m pip install ftfy regex ipdb fvcore
```

CLIP pretrained weights are downloaded automatically on first use and cached in `~/.cache/clip`.

## Data

Prepare HMDB51, UCF101, or Kinetics-100 videos separately. Configurations and split lists are under [configs/projects/SF-CLIP](configs/projects/SF-CLIP).

- Set `DATA.DATA_ROOT_DIR` to the video root. Video paths in the split lists are relative to this directory.
- Set `DATA.ANNO_DIR` to the corresponding split-list directory. The current loader reads `train_few_shot.txt` for training and `test_few_shot.txt` for evaluation, including evaluation during training.
- Split-list entries use `train<class_id>//<video_path>` or `test<class_id>//<video_path>`. Entries in `TRAIN.CLASS_NAME` and `TEST.CLASS_NAME` must match the zero-based class IDs in their respective lists. Each entry contains an ordered stage description, such as `stage one then stage two`.

Stage descriptions are already included in the YAML configurations; training and evaluation do not require an online language-model call.

## Usage

Run commands from the repository root. Edit data paths, `NUM_GPUS`, `TRAIN.BATCH_SIZE`, and `TEST.BATCH_SIZE` in the YAML for your setup. Set boolean and numeric options directly in YAML: command-line overrides currently retain these values as strings.

### Training

Kinetics-100, CLIP ViT-B/16, 5-way 1-shot:

```bash
python runs/run.py --cfg configs/projects/SF-CLIP/kinetics100/SF-CLIP_K100_ViT-B16_5way1shot_v1.yaml
```

### Checkpoint evaluation

Copy the training configuration to `eval_5way1shot.yaml` in the same directory so its relative `_BASE` path remains valid. Update the existing keys below while preserving the remaining configuration. Replace the checkpoint placeholder with a trained checkpoint matching the selected model and dataset, and use a separate output directory.

```yaml
TRAIN:
  ENABLE: false
TEST:
  ENABLE: true
  CHECKPOINT_FILE_PATH: ./path/to/checkpoint.pyth
OUTPUT_DIR: ./output/k100_eval
```

```bash
python runs/run.py --cfg configs/projects/SF-CLIP/kinetics100/eval_5way1shot.yaml
```

Evaluation uses `TRAIN.WAT_TEST` and `TRAIN.SHOT_TEST` for the number of classes and support videos per class, and `TRAIN.NUM_TEST_TASKS` for the episode count.
