# GAN Augmented CNN

This repos hosts the code used to artificially extend the [Hyper-Kvasir dataset](https://datasets.simula.no/hyper-kvasir/) 
in order to improve classification performance.

## Notebooks

Notebooks containing article's experiences are provided in folder `notebooks` to ease reproducibility.

## Create training dataset

Training and validation set needs to be created with the script `generate_dataset.py`. It aims to create a dataset from
image folders that take care of the format needed by training scripts :

```
output_folder
├── train 
│   ├── class1
│   └── class2
└── val 
    ├── class1
    └── class2
```

| Argument name   | Expected value                                                                                                                             |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| checkpoints     | List of string contains paths to SG2 checkpoints (optional if generate_number = 0  or one per class defined in class_names)                |
| class_folders   | String dictionary based on the following format (Keys must be the same as class_names): '{"class_name": ["folder1", ...]}'                 |
| generate_number | With dataset equalisation: #(Real Images) + #(Synthetic image) = generate_number, Without: #(Synthetic image) = generate_number            |
| output_dir      | Output directory                                                                                                                           |
| split_file      | CSV file describing (file-name;class-name;split-index) training (0) and validation sets (!=0).                                             |

The following commands generates the training set for the experimental protocol CUSTOM-UC.
```shell
%run generate_dataset.py \
    --checkpoints 'PATH_TO_CHECKPOINTS/2.sg2ada_non_pathological.pkl' 'PATH_TO_CHECKPOINTS/3.sg2ada_pathological.pkl' \
    --class_folders '{"non_pathological": ["HK/lower-gi-tract/lgi-quality-of-mucosal-views/bbps-2-3", "HK/lower-gi-tract/lgi-pathological-findings/uc-grade-1"], "pathological": ["HK/lower-gi-tract/lgi-pathological-findings/uc-grade-2", "HK/lower-gi-tract/lgi-pathological-findings/uc-grade-3"]}'
    --generate_number 0 --output_dir ../custom_uc_raw --split_file ../training-set-full/splits/both_2_fold_split.csv
```

The following commands generates a training set for the whole Hyper-Kvasir dataset.
```shell
%run generate_dataset.py --checkpoints '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' '' \
  --class_folders '{"cecum": ["../training-set-full/lower-gi-tract/lgi-anatomical-landmarks/cecum"], "ileum": ["../training-set-full/lower-gi-tract/lgi-anatomical-landmarks/ileum"], "retroflex-rectum": ["../training-set-full/lower-gi-tract/lgi-anatomical-landmarks/retroflex-rectum"], "hemorrhoids": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/hemorrhoids"], "polyps": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/polyps"], "ulcerative-colitis-grade-0-1": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/uc-grade-1/ulcerative-colitis-grade-0-1"], "ulcerative-colitis-grade-1": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/uc-grade-1/ulcerative-colitis-grade-1"], "ulcerative-colitis-grade-1-2": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/unused-uc/ulcerative-colitis-grade-1-2"], "ulcerative-colitis-grade-2": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/uc-grade-2/ulcerative-colitis-grade-2"], "ulcerative-colitis-grade-2-3": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/uc-grade-3/ulcerative-colitis-grade-2-3"], "ulcerative-colitis-grade-3": ["../training-set-full/lower-gi-tract/lgi-pathological-findings/uc-grade-3/ulcerative-colitis-grade-3"], "bbps-0-1": ["../training-set-full/lower-gi-tract/lgi-quality-of-mucosal-views/bbps-0-1"], "bbps-2-3": ["../training-set-full/lower-gi-tract/lgi-quality-of-mucosal-views/bbps-2-3"], "impacted-stool": ["../training-set-full/lower-gi-tract/lgi-quality-of-mucosal-views/impacted-stool"], "dyed-lifted-polyps": ["../training-set-full/lower-gi-tract/lgi-therapeutic-interventions/dyed-lifted-polyps"], "dyed-resection-margins": ["../training-set-full/lower-gi-tract/lgi-therapeutic-interventions/dyed-resection-margins"], "pylorus": ["../training-set-full/upper-gi-tract/ugi-anatomical-landmarks/pylorus"], "retroflex-stomach": ["../training-set-full/upper-gi-tract/ugi-anatomical-landmarks/retroflex-stomach"], "z-line": ["../training-set-full/upper-gi-tract/ugi-anatomical-landmarks/z-line"], "barretts": ["../training-set-full/upper-gi-tract/ugi-pathological-findings/barretts"], "barretts-short-segment": ["../training-set-full/upper-gi-tract/ugi-pathological-findings/barretts-short-segment"], "esophagitis-a": ["../training-set-full/upper-gi-tract/ugi-pathological-findings/esophagitis-a"], "esophagitis-b-d": ["../training-set-full/upper-gi-tract/ugi-pathological-findings/esophagitis-b-d"] }'\
  --generate_number 0 --output_dir ../fullhk_dataset_raw --split_file ../training-set-full/splits/both_2_fold_split.csv
```

## Training

### CNN

The `main.py` script contains all classification trainings.

| Argument name  | Expected value                                                                      |
|----------------|-------------------------------------------------------------------------------------|
| dataset        | Training dataset folder path                                                        |
| input_size     | Image size (Default = 256)                                                          |
| da             | Enable basic data augmentation (Default = True)                                     |
| lr             | Learning rate value (Default = 0.001)                                               |
| clr            | Enable cyclic learning rate (Default = True)                                        |
| output_dir     | Output directory                                                                    |
| batch_size     | Batch size (Default = 256)                                                          |
| epoch_number   | Epoch number (Default = 30)                                                         |
| pretrained     | Load pretrained densenet (ImageNet or path to the checkpoint, Default = '')         |
| save_best      | Save each model epoch that outperforms the previous ones                            |
| continue_train | Continue training from checkpoint (Default = '')                                    |
| architecture   | Classifier architecture (Default = 'densenet161', see `network/cnn/classifiers.py`) |

With this script, the model will be trained with SGD (momentum = 0.9) and based on a Cross Entropy Loss. Tensorboard
files will be output in the `output_dir/runs/` folder.

Example :

```shell
%run main.py --batch_size 128  --dataset ../fullhk_dataset_raw --da True --output_dir ./ResNet50/FHKRawNoPretrain --architecture resnet50
```

```shell
%run main.py --batch_size 64  --dataset ../custom_uc_raw --da True --output_dir ./DenseNet161/CUCRawImageNet --architecture densenet161 --pretrained ImageNet
```

All metrics are output in a `results.json` with the following format:

```json
{
  "cm": [[0.0, 0.0], [0.0, 0.0]],
  "macro": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "micro": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  },
  "MCC": 0.0
}
```

### StyleGAN2

`dnnlib` and `torch_utils` folders are mandatory to enable the support of 
[StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch).

StyleGAN2 weights can be created and trained with the following repos [NVLabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

Using `generate_dataset.py`:

```shell
%run generate_dataset.py --checkpoints '' '' \
  --class_folders '{"non_pathological": ["../training-set-full/lower-gi-tract/lgi-quality-of-mucosal-views/bbps-2-3", "../training-set-full/lower-gi-tract/lgi-pathological-findings/uc-grade-1"]}'\
  --generate_number 0 --output_dir ../preprocessed_dataset --split_file ../training-set-full/splits/both_2_fold_split.csv

%cd ..

from distutils.dir_util import copy_tree
import os
os.mkdir('./non-pathological')
copy_tree('./preprocessed_dataset/train/non_pathological', './non-pathological')

!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

%cd stylegan2-ada-pytorch

dataset = '../non_pathological.zip'
!python dataset_tool.py --source ../non_pathological --dest $dataset --width 256 --height 256

!pip install ninja

output_dir = '.'

import os
checkpoint = os.path.join(output_dir, '1.sg2ada_unlabeled_pt.pkl')

!python train.py --outdir=$output_dir --data=$dataset --aug=ada --snap=10 --freezed=3 --resume=$checkpoint --workers=2 --mirror=1
```

## Hyper Kvasir

To paper results use a slightly modified version of Hyper-Kvasir in order to avoid name clash. A modified split file
is provided within the `samples` directory and needs to be placed in the `splits` folder of the dataset. These 
modifications only concerns naming and does not involve merging or modify the data of the dataset. The modified dataset 
follows the next structure:

```
├── lower-gi-tract
│   ├── lgi-anatomical-landmarks
│   │   ├── cecum
│   │   ├── ileum
│   │   └── retroflex-rectum
│   ├── lgi-pathological-findings
│   │   ├── hemorrhoids
│   │   ├── polyps
│   │   ├── uc-grade-1
│   │   │   ├── ulcerative-colitis-grade-0-1
│   │   │   └── ulcerative-colitis-grade-1
│   │   ├── uc-grade-2
│   │   │   └── ulcerative-colitis-grade-2
│   │   ├── uc-grade-3
│   │   │   ├── ulcerative-colitis-grade-2-3
│   │   │   └── ulcerative-colitis-grade-3
│   │   └── unused-uc
│   │       └── ulcerative-colitis-grade-1-2
│   ├── lgi-quality-of-mucosal-views
│   │   ├── bbps-0-1
│   │   ├── bbps-2-3
│   │   └── impacted-stool
│   └── lgi-therapeutic-interventions
│       ├── dyed-lifted-polyps
│       └── dyed-resection-margins
├── splits
│   └── hk_2_fold_split_with_paths.csv
├── upper-gi-tract
│   ├── ugi-anatomical-landmarks
│   │   ├── pylorus
│   │   ├── retroflex-stomach
│   │   └── z-line
│   └── ugi-pathological-findings
│       ├── barretts
│       ├── barretts-short-segment
│       ├── esophagitis-a
│       └── esophagitis-b-d
└── wce-crohn-ipi
    ├── wce-normal
    └── wce-pathological
        ├── wce-apthoid-ulceration
        ├── wce-edama
        ├── wce-erythema
        ├── wce-stenosis
        ├── wce-ulceration-between-3mm-10mm
        └── wce-ulceration-over-10mm
```

# References

```
H. Borgli, et al,  HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy, Scientific Data 7 (2020) 283.doi:10.1038/s41597-020-00622-y.
```

```
T. Karras, et al, Analyzing and improving the image quality of StyleGAN, in: CVPR, 2020. doi:10.1109/CVPR42600.2020.00813.
```

```
T. Karras, et al, Training generative adversarial networks with limited data, in: NeurIPS, 2020.
```
