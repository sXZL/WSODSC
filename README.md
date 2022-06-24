# Weakly Supervised Object Detection with Symmetry Context

This is a Pytorch implementation of our WSODSC. 


## Introduction
WSODSC is a framework for weakly supervised object detection with deep ConvNets.

Our code is written based on [PyTorch](https://pytorch.org/) and [wetectron](https://github.com/NVlabs/wetectron).

Sincerely thanks for your recources.

## Installation
### Requirements:
- Python 3
- Pytorch 1.5+
- torchvision 

We follow the same installation steps with [wetectron](https://github.com/NVlabs/wetectron).

- Check [INSTALL.md](https://github.com/NVlabs/wetectron/blob/master/docs/INSTALL.md) for installation instructions.


## Datasets
For PASCAL VOC 2007 dataset, [download](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and then


```Shell
mkdir -p datasets/voc
ln -s /path_to_VOCdevkit/VOC2007 datasets/voc/VOC2007
```


## Proposals
Download the proposals from [Google-drive](https://drive.google.com/drive/folders/11xiHM1P65VXIa80zJb_OSjEtvIS7oc2L?usp=sharing) first,


##### Description 
 - selective\_search\_data: precomputed proposals of VOC 2007
 - context\_proposals: precomputed context proposals
 - pretrained\_models

```Shell
mkdir proposal
ln -s  /path/to/downloaded/files/*.pkl proposal/
```
 
 

## Training
```bash
export NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
    --config-file "configs/ctx/mist-reg-ctx.yaml" --use-tensorboard \
    OUTPUT_DIR /path/to/output/dir
```

## Evaluation
The pretrained models can be found at [Google-drive](https://drive.google.com/drive/folders/11xiHM1P65VXIa80zJb_OSjEtvIS7oc2L?usp=sharing).
Note that all these model were trained using 1 Nvidia V100 GPU(32GB) and Pytorch 1.7.

## voc models

| Train data        | Eval data        | Config                       | Backbone     | mAP    |
|:------------------|------------------|------------------------------|--------------|-------:|
| voc 2007          | voc 2007 test    | ctx/mist-reg-ctx.yaml          | VGG-16       | 54.2   |
| voc 2012          | voc 2012 test    | voc/mist-reg-ctx-voc12.yaml        | VGG-16       | 48.3   |

Here is an example to evaluate the released model:

```bash
export NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
    --config-file "configs/ctx/mist-reg-ctx.yaml" TEST.IMS_PER_BATCH 8 \
    OUTPUT_DIR /path/to/output/dir \
    MODEL.WEIGHT /path/to/model
```

## License

This code is released under the [Nvidia Source Code License](LICENSE). 

This project is built upon [wetectron](https://github.com/facebookresearch/maskrcnn-benchmark), which is released under [Nvidia Source Code License](LICENSE).


