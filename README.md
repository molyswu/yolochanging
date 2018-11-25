# Install dataset and environ
- cd pascalvoc/  
- bash download.sh  to download dataset(500 MB, sorry I only downlodaed one year data, but it is this big)
- conda py36tf_1.yml


# Aim
- stable 99%-100% GPU usage for 2 tests.

#  Run 2 tests
### Single machine training 
bash run_single.sh --watch_gpu=0 [--watch-gpu should be the same as the visible gpu]
### Distributed training  
bash ps.sh  
bash worker0.sh --watch-gpu=0  
bash worker1.sh  --watch-gpu=1  
[--watch-gpu should be the same as the visible gpu]

# yolo/config.py
- Change the parameters in yolo_tensorflow/yolo/config.py  
#-----------------Parameters to be changed-----------------------------#
#############################################################################
PS_HOSTS  = '172.20.83.210:8897'

WORKER_HOSTS = '172.20.83.210:8898,172.20.83.202:8898'

BATCH_SIZE = 10

NUM_ENQUEUE_THREADS = 2

MUL_QUEUE_BATCH = 2

PROFILER_SAVE_STEP = 60

SUMMARY_SAVE_STEP = 120

#############################################################################
- MUL_QUEUE_BATCH means how many times is the queue size is of batch_size. 2 means 10*2=20
- PROFILER_SAVE_STEP, SUMMARY_SAVE_STEP can not be too small, because I found especially summary take long time
