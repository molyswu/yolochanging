# Install dataset and environ
- cd pascalvoc/  
- bash download.sh  to download dataset(500 MB, sorry I only downlodaed one year data, but it is this big)
- pip install requirement.txt  


# Aim
- stable 99%-100% GPU usage for benchmarks.
- better speed for distribtued trainnig than single machine training 

#  Run 2 benchmarks
### Single machine training 
run run_single.sh, please write the --watch-gpu the same id as the visible gpu
### Distributed training  
run ps.sh  
run worker0.sh --watch-gpu=0  
run worker1.sh  --watch-gpu=1  
set watch-gpu id the same as visible gpu to each one. There could be a problem that some can always see first gpu.
It should have been solved by adding in the code ""

# How: Change parameters 
- Change the parameters in yolo_tensorflow/yolo/config.py  
- change hosts and workers to start, BATCH_SIZE,NUM_ENQUEUE_THREADS, MUL_QUEUE_BATCH(how many times is the queue size is based on batch_size, eg. 2 means double the size of batchsize as the size of inputqueue, 2 or 3 should be enough) and PROFILER_SAVE_STEP, SUMMARY_SAVE_STEP to reach 100% of GPU utilisation and cpu utisage.
