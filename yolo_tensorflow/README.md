## YOLO\_tensorflow
This contains 1. base\_rawinput. 2. threadlevel parallelism inputpipeline 3. debug validation 4. PS communication module distributed training.  
This is synchronized with Yolo_Tensorflow_InputPipeline

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test phase.

### Installation

1. Clone yolo_tensorflow repository
	```Shell
	$ git clone https://github.com/hizhangp/yolo_tensorflow.git
    $ cd yolo_tensorflow
	```

2. Download Pascal VOC dataset, and create correct directories
	```Shell
	$ ./download_data.sh
	```

3. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
weight file and put it in `data/weight`

4. Modify configuration in `yolo/config.py`

5. Training
	```Shell
	$ python train.py
	```

6. Test
	```Shell
	$ python test.py
	```

### Requirements
1. Tensorflow

2. OpenCV
