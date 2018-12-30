# Branch Readme 
In this horovod test, base test is fixed with warm-up and good local and global fps  


- cd pascalvoc/  
- bash download.sh  to download dataset(500 MB, sorry I only downlodaed one year data, but it is this big)
- (For environment, I only used tensorflow-gpu and opencv. If your environment does not work, try this). 
- cd env, bash build_env.sh


# Aim
- stable 99%-100% GPU usage for 2 tests.

#  Run 2 tests
### 1. Single machine training 
bash run_single.sh --watch_gpu=0 [--watch-gpu should be the same as the visible gpu]
### 2. Distributed training  
bash ps.sh  
bash worker0.sh --watch-gpu=0  
bash worker1.sh  --watch-gpu=1  
[--watch-gpu should be the same as the visible gpu]

# Change yolo_tensorflow/yolo/config.py
- Change parameters in yolo_tensorflow/yolo/config.py  
- For test1, just ignore PS_HOSTS and WORKER_HOSTS. Only change it for test2
- MUL_QUEUE_BATCH means how many times is the queue size is of batch_size. 2 means 10*2=20
- PROFILER_SAVE_STEP, SUMMARY_SAVE_STEP can not be too small, because I found especially summary take long time
- How does the yolo/config.py look like and where to change? change here  

#############################################################################  
PS_HOSTS  = '172.20.83.210:8897'  

WORKER_HOSTS = '172.20.83.210:8898,172.20.83.202:8898'  

BATCH_SIZE = 10  

NUM_ENQUEUE_THREADS = 2  

MUL_QUEUE_BATCH = 2  

PROFILER_SAVE_STEP = 60  

SUMMARY_SAVE_STEP = 120  

#############################################################################  
