# Remote Sensing Reader (rsreader)

This package provides data loader for semantic segmentation of aerial images.
By now, it supports gluon/mxnet and pytorch.
And it can be used to the tasks of the regular semantic segmentation and
semantic segmentation based on multi-modal data.

## Install
> Step1: generate whl package
 - python3 setup.py bdist_wheel

> Step2: install
 - cd dist
 - pip3 install xxx.whl

## Usage

> Step 1: list all the aerial images in a file:

- List file (.lst) for a regular semantic segmentation task

For the task of regular semantic segmentation, the list file can
be seen as follows.
```angular2html
irrg/top_mosaic_09cm_area1.tif gt/top_mosaic_09cm_area1.tif
irrg/top_mosaic_09cm_area13.tif gt/top_mosaic_09cm_area13.tif
irrg/top_mosaic_09cm_area17.tif gt/top_mosaic_09cm_area17.tif
irrg/top_mosaic_09cm_area21.tif gt/top_mosaic_09cm_area21.tif
irrg/top_mosaic_09cm_area23.tif gt/top_mosaic_09cm_area23.tif
irrg/top_mosaic_09cm_area26.tif gt/top_mosaic_09cm_area26.tif
irrg/top_mosaic_09cm_area3.tif gt/top_mosaic_09cm_area3.tif
irrg/top_mosaic_09cm_area32.tif gt/top_mosaic_09cm_area32.tif
irrg/top_mosaic_09cm_area37.tif gt/top_mosaic_09cm_area37.tif
irrg/top_mosaic_09cm_area5.tif gt/top_mosaic_09cm_area5.tif
irrg/top_mosaic_09cm_area7.tif gt/top_mosaic_09cm_area7.tif
```

The first column refers to the true orthophoto and the second
column refers to the ground truth. If the ground truth is not provided,
the second column can be deleted.

- List file (.lst) for semantic segmentation based on multi-modal data

For the task of semantic segmentation based on multi-modal data,
the list file can be seen as follows.

```angular2html
irrg/top_mosaic_09cm_area1.tif dsm/dsm_09cm_matching_area1.tif gt/top_mosaic_09cm_area1.tif
irrg/top_mosaic_09cm_area13.tif dsm/dsm_09cm_matching_area13.tif gt/top_mosaic_09cm_area13.tif
irrg/top_mosaic_09cm_area17.tif dsm/dsm_09cm_matching_area17.tif gt/top_mosaic_09cm_area17.tif
irrg/top_mosaic_09cm_area21.tif dsm/dsm_09cm_matching_area21.tif gt/top_mosaic_09cm_area21.tif
irrg/top_mosaic_09cm_area23.tif dsm/dsm_09cm_matching_area23.tif gt/top_mosaic_09cm_area23.tif
irrg/top_mosaic_09cm_area26.tif dsm/dsm_09cm_matching_area26.tif gt/top_mosaic_09cm_area26.tif
irrg/top_mosaic_09cm_area3.tif dsm/dsm_09cm_matching_area3.tif gt/top_mosaic_09cm_area3.tif
irrg/top_mosaic_09cm_area32.tif dsm/dsm_09cm_matching_area32.tif gt/top_mosaic_09cm_area32.tif
irrg/top_mosaic_09cm_area37.tif dsm/dsm_09cm_matching_area37.tif gt/top_mosaic_09cm_area37.tif
irrg/top_mosaic_09cm_area5.tif dsm/dsm_09cm_matching_area5.tif gt/top_mosaic_09cm_area5.tif
irrg/top_mosaic_09cm_area7.tif dsm/dsm_09cm_matching_area7.tif gt/top_mosaic_09cm_area7.tif
```
The first column refers to the true orthophoto, 
the second refers to the Digital Surface Model (DSM),
and the third column refers to the ground truth.
If the ground truth is not provided,
the last column can be deleted.

> Step 2: define the netreader

Define a netreader based on your task.
By now, we provide the common SegReader, DeeplabReader, SCNNReader and so on,
which can be found in the directory of 'rsreader.netreader'.

```angular2html
from rsreader.utility.Normalization import *
from rsreader.utility.DataAugmentation import *
from rsreader.utility.ImageTrans import ImageTrans
from rsreader.utility.JointTrans import JointTrans

joint_trans = JointTrans([])
img_trans = ImageTrans([])
gt_trans = ImageTrans(trans=[])
netreader = SegReader(flist_name='/path/to/your/list_file/trainset.lst',
                      data_root='/path/to/your/data_root',
                      batchsize=1, cropsize=224, step=224,
                      img_trans=img_trans, gt_trans=gt_trans, joint_trans=joint_trans,
                      withgt=True, bandlist=None, sampleseed=-1,
                      lvreadertype='gdal', parsertype='common', openfirstly=True)
```

> Step 3: Define a Dataset.

Define a Dataset based on your favourite deep learning framework.
We have an example based on gluon/mxnet.

```angular2html
train_set = GluonDataset(netreader, niter=-1, mode='shuffle')
```

## Extension
This package has a good extension character.
You can define your own class from the low-level reader, parsers
and netreader.
The documents of lvreader and store are shown in the README.md 
in the corresponding directory.

#### Citation
Welcome to use our codes.

We would be very glad if you can cite our papers:
```angular2html
@article{chen2018semantic,
  title={SEMANTIC SEGMENTATION OF AERIAL IMAGERY VIA MULTI-SCALE SHUFFLING CONVOLUTIONAL NEURAL NETWORKS WITH DEEP SUPERVISION.},
  author={Chen, Kaiqiang and Weinmann, Michael and Sun, Xian and Yan, Menglong and Hinz, Stefan and Jutzi, Boris and Weinmann, Martin},
  journal={ISPRS Annals of Photogrammetry, Remote Sensing \& Spatial Information Sciences},
  volume={4},
  number={1},
  year={2018}
}
```
and
```angular2html
@article{chen2018semantic,
  title={Semantic segmentation of aerial images with shuffling convolutional neural networks},
  author={Chen, Kaiqiang and Fu, Kun and Yan, Menglong and Gao, Xin and Sun, Xian and Wei, Xin},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={15},
  number={2},
  pages={173--177},
  year={2018},
  publisher={IEEE}
}
```