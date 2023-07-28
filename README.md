# TransferSAM

[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description
This repository attempts to optimise Segment Anything Model (SAM vit_b) developed by Meta AI for medical image segmentation, namely **[Kidney and Kidney Tumor Segmentation](https://kits-challenge.org/kits23/)** using transfer learning techniques, such as: 
    
    - Feature Extraxction
    - Fine-Tuning
    - Domain Adaption

It was developed as part of the Applied Deep Learning with TensorFlow and PyTorch course conducted at LMU Munich under the guidance and supervision of Mina Rezaei.

## Key Features
- Generate High-quality image embeddings using Image Encoder of the pre-trained SAM
- Fine-Tune (or) Retrain SAM's Lightweight Mask Decoder for specific tasks and domains, enhancing the model's performance for medical image segmentation

## Note
The `segment_anything` directory has been derived from the official **[Segment Anything](https://github.com/facebookresearch/segment-anything)** repository.
It contains modified versions of the original `build_sam.py` and `/modeling/mask_decoder.py` scripts.
These modifications have been clearly marked with the comment: `# Modified by: https://github.com/Noza23/TransferSAM`. These changes were made to adapt the code to the specific requirements and objectives of this project.

## Installation
The requirements are listed in the **[setup.py](https://github.com/Noza23/TransferSAM/blob/main/setup.py)**

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
In the first place download the **[KiTSAM.pkl](https://github.com/Noza23/TransferSAM)**
Afterwards, follow the code:

```
from TransferSAM import Predictor
with open('<path/to/KiTSAM.pkl>', 'rb') as file:
    kitsam = pickle.load(file)
predictor = Predictor(kitsam.model_roi, kitsam.tumor_decoder, kitsam.cyst_decoder)
prediction_mask = predictor.predict_complete(<your image_slice of shape HxW>)
```

or generate predictions for an entire 3D case of shape SxHxW

```
prediction_masks = predictor.predict_case(<your 3D image of shape SxHxW>)
```

## Model Training
### Dataset
To download the dataset used in training follow **[Data Download](https://github.com/neheller/kits23#data-download)** instructions of **[kits23](https://github.com/neheller/kits23)** repository.

### Generate Embeddings
In order to generate the embeddings for KiTS Dataset run the following script:
```
python3 scripts/embed.py -i <Path to the dataset directory>  --checkpoint <SAM checkpoint> --max_size <int> --batch_size <int> --case_start <int> --case_end <int>
```
- SAM checkpoints can be download from **[segment-anything](https://github.com/facebookresearch/segment-anything/blob/main/README.md#model-checkpoints)**

Generated Embeddings for each case will be saved in each case directory of the dataset.

### Start Training
In order to start training, fill out the **config.yaml** and execute the following script:
```
python3 scripts/train.py --config_file config.yaml --device "cuda:0"
```

### Combining models
In the Project 3 different Mask Decoders were trained:
1. Region of Interst (ROI) model for identifying regions of interest (Kidneys) in the image.
2. Tumor model for identifying Kidney Tumors in the image.
3. Cyst model for identifying Kidney Cysts in the image.

In the end the 3 models were combined into a single class instance which was stored in a pickle format using the script:
```
python3 scripts/build_KiTSAM.py --roi_model <Path to ROI model> --tumor_model <Path to tumor model> --cyst_model <Path to cyst model>
```

### Prediction
In order to generate predictions for multiple new **.nii.gz** images put them all in one directory and execute following script:
 ```
python3 scripts/predict.py --casesdir <Directory of new cases> --case_start <int> --case_end <int> --output_path <Output Directory> --device "cuda:0"
```

## Acknowledgements
- Meta AI for **[Segment Anything](https://github.com/facebookresearch/segment-anything)** repository.
- Organisers of the **[KiTS23](https://kits-challenge.org/kits23)** challange for making the **[KiTS Dataset](https://github.com/neheller/kits23/tree/main/dataset)**  publicly avaliable.