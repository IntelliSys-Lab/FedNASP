# FedNASP

This repository provides the implementation of **FedNASP**, a step-wise personalized federated learning framework for Vision-Language Navigation (VLN).

> Federated learning (FL) protects sensitive (Vision-Language Navigation) VLN data without centralizing trajectories or instructions, but severe non-IID environments make personalized FL (pFL) necessary.Moreover, VLN poses several coupled challenges for personalized federated learning, including environment heterogeneity, multimodal language-vision fusion, and long-horizon navigation with time-varying decision contexts. To address these challenges, we propose FedNASP, a step-wise personalized federated learning framework for VLN. The key idea is to dynamically calibrate personalization strength along a navigation trajectory. Specifically, we introduce a lightweight Step-wise Personalized Modulator (SPM) that predicts personalization strength at each navigation step. We further design a structure-aware adapter-based personalized prefix injection mechanism that enables client-specific grounding while keeping the backbone shared across clients.

## Repository Layout

- `reverie_src/`: REVERIE training and evaluation code
- `cvdn_src/`: CVDN training and evaluation code
- `connectivity/`: Matterport connectivity graphs used by both tasks
- `requirements.txt`: Python environment dependencies

## 1. Install Matterport3D Simulator

Follow the official Matterport3D Simulator instructions here:

- <https://github.com/peteanderson80/Matterport3DSimulator>

This repository already contains the simulator source/build files needed by the current codebase. After building the simulator, make sure the `MatterSim` Python module is importable in your environment.

## 2. Download Data and Checkpoints

This repository ships code only. Please download the required data, image features, and pretrained weights separately.

### REVERIE

Download REVERIE-related data and model weights from:

- <https://github.com/airbert-vln/airbert-recurrentvln>

At minimum, the REVERIE code paths expect assets under repository-relative locations such as:

```text
datasets/REVERIE/annotations/
datasets/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin
img_features/ResNet-152-places365.tsv
img_features/REVERIE_obj_feats.pkl
```

### CVDN

Download CVDN-related data and checkpoints from:

- <https://www.dropbox.com/scl/fo/nsitfiyh5taz4xg4nxj7b/AOqFG0t3ZrReqJ_m3qP5Z8k?rlkey=e0wd5c9a1mhbvfd96lb7d30xx&e=1&dl=0>

The CVDN code expects data under repository-relative locations such as:

```text
datasets/CVDN/annotations/
img_features/ResNet-152-places365.tsv
```

Notes:

- `connectivity/` is already included in this repository.
- Small config/vocab files that are safe to version are already included, such as `datasets/CVDN/train_vocab.txt`, `datasets/CVDN/trainval_vocab.txt`, and `datasets/vln-bert/bert_base_6_layer_6_connect.json`.
- If you use alternative backbones or checkpoints, place them under the relative paths expected by the corresponding scripts.

## 3. Environment Setup

Use the repository `requirements.txt` as the primary environment specification:

```bash
pip install -r requirements.txt
```

## 4. Run Scripts

### REVERIE

```bash
bash reverie_src/run_ours.bash
```

### CVDN

```bash
bash cvdn_src/run_ours.bash
```

## 5. Outputs

Training scripts will create output directories such as:

- `logs/`
- `snap/`
- `plots/`

These directories are runtime artifacts and are intentionally excluded from version control.