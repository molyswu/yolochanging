# Install
- cd pascalvoc/  
- pip install requirement.txt  
- bash download.sh  to download dataset(~ 50 MB)

# Befoer Running  
- Change the parameters in yolo_tensorflow/yolo/config.py  
- change hosts and workers to start.
- change BATCH_SIZE,NUM_ENQUEUE_THREADS, MUL_QUEUE_BATCH(how many times is the queue size is based on batch_size, eg. 2 means double the size of batchsize as the size of inputqueue ) and PROFILER_SAVE_STEP, SUMMARY_SAVE_STEP to reach 100% of GPU utilisation and cpu utisage.

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
## About the save steps of profiler
- Save steps of profiling could be 60, but not too small. Save summary rarely.
## Watch gpu may be left running
nvidia-smi and cpu log start during running, and will be killed with Ctrl+C keyboard interruption or the training end itself normally.  
If the training process end because of lack of memory. nvidia-smi has to be killed manually

# Part of benchmarks best parameter settings
Pacalvoc must be downloaded otherwise 0 feeded. 
queue_size seems can not be too big. I set queuesize to 2 and get bad
Worst is frequent saving profiler and summary take a lot of memory
