# Weakly Supervised Object Detection with Symmetry Context

This is a Pytorch implementation of our WSODSC. 


## Introduction
WSODSC is a framework for weakly supervised object detection with deep ConvNets.

Our code is written based on [PyTorch](https://pytorch.org/) and [wetectron](https://github.com/NVlabs/wetectron).

Sincerely thanks for your recources.


## Additional resources
[Google-drive](https://drive.google.com/drive/u/2/folders/1DYKIOrM0X3o_kdA-p932XYcIzku2fKAM)

##### Description 
 - selective\_search\_data: precomputed proposals of VOC 2007
 - context\_proposals: precomputed context proposals


## Prepare
#### Installation

We follow the same installation steps with [wetectron](https://github.com/NVlabs/wetectron).

Check [INSTALL.md](docs/INSTALL.md) for installation instructions.
Check [GETTING_STARTED](docs/GETTING_STARTED.md) for detailed instrunctions. 


#### Datasets
For example, PASCAL VOC 2007 dataset


1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    	cd $FRCN_ROOT/data
    	ln -s $VOCdevkit VOCdevkit2007
    	```


## License

This code is released under the MIT License (refer to the LICENSE file for details). 

