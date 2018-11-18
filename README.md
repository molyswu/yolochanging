# Install
- cd pascalvoc/  
- pip install requirement.txt  
- bash download.sh  to download dataset(less than 500 MB)

# Running  
- Change the parameters in yolo_tensorflow/yolo/config.py  
- change hosts to start.
- change batchsize,NUMENEUQUETHREADS, MulOfQueuesizeToBatchsize to reach 100% of GPU utilisation and cpu utisage

# Naive version feed input:  
run base_run.sh

# Single machine training with pipeline  
run single_run.sh, please write the --watch-gpu the same as the visible gpu


# Distributed training  
run ps.sh
run worker0.sh --watch-gpu=0
run worker1.sh  --watch-gpu=1
set different visible gpu to each one. There could be a problem that some can always see first gpu.
It should be solved by adding in the code

increase the batch size and queue size. watch the profiler json so that gpu is fully used and cpu do not be idle for too long time

# Notice
nvidia-smi and cpu log will start during running, and will be killed with Ctrl+C or the training end itself.  
If the training end because of lack of memory. nvidia-smi has to be killed manually

# Part of benchmarks best parameter settings
Pacalvoc must be downloaded otherwise the loss will drop very fast. it just feed 0.  
queue_size seems can not be too big. I set queuesize to 2 and get bad
