# CIL Project Based on Mask RCNN
## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_balloon.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).

## Run Jupyter notebooks
1. Jupyter notebooks are not customized for CIL project.

## Train the CIL road segmentation model

Train a new model starting from pre-trained COCO weights
```
python3 cil.py train --dataset=/path/to/cil/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 cil.py train --dataset=/path/to/cil/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 cil.py train --dataset=/path/to/cil/dataset --weights=imagenet
```

The training settings are included in `CilConfig(Config)` class in the code `cil.py`.
Update the schedule to fit your needs.
