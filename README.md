# TransferSAM

[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description
This repository enables users to perform transfer learning techniques, such as: Feature Extraxction, Fine-Tuning and Domain Adaption from the new SAM model developed by Meta AI.
It was developed as part of the Applied Deep Learning with TensorFlow and PyTorch course conducted at LMU Munich under the guidance and supervision of Mina Rezaei.

## Key Features
- Generate High-quality image embeddings using Image Encoder of the pre-trained SAM
- Fine-Tune SAM's Lightweight Mask Decoder for specific tasks and domains, enhancing the model's performance for image segmentation
- Retrain SAM's Lightweight Mask Decoder from scratch
- Extend SAM to multi-instance segmentation

## Note
The `segment_anything` directory included in this project has been derived from the official **[Segment Anything](https://github.com/facebookresearch/segment-anything)** repository. It contains modified versions of the original `build_sam.py` and `/modeling/mask_decoder.py` scripts. These modifications have been clearly marked with the comment: `# Modified by: https://github.com/Noza23/TransferSAM`. These changes were made to adapt the code to the specific requirements and objectives of this project.


## Installation
The requirements are listed in the **[setup.py](https://github.com/Noza23/TransferSAM)**

You can install TransferSAM using `pip`:

```
pip install git+https://github.com/Noza23/TransferSAM.git
```

or clone the repository locally and install with:

```
git clone git@github.com:Noza23/TransferSAM.git
cd TransferSAM; pip install -e .
```

## Usage
In the first place download desired checkpoint from **[Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)** (In the examples above, we have used **vit_b** - smallest SAM checkpoint)