# Install
- cd pascalvoc/  
- pip install requirement.txt  
- bash download.sh  to download dataset(500 MB, sorry I only downlodaed one year data, but it is this big)

# Aim
- train 3 benchmarks: naive version, single machine with pipeline, distirbuted training. fps(frames per second) should rise.
- achieve stable 99%-100% GPU usage for last two, achieve 70%-80% GPU usage for naive version. 

# How: Change parameters to reach 100% GPU usage  
- Change the parameters in yolo_tensorflow/yolo/config.py  
- change hosts and workers to start.
- change BATCH_SIZE,NUM_ENQUEUE_THREADS, MUL_QUEUE_BATCH(how many times is the queue size is based on batch_size, eg. 2 means double the size of batchsize as the size of inputqueue, 2 or 3 should be enough) and PROFILER_SAVE_STEP, SUMMARY_SAVE_STEP to reach 100% of GPU utilisation and cpu utisage.
- loss is not important, better do not save checkpoint. You may stop (CTRL+C) after 5 minutes of training. Just wait it get stable fps.


#  Set of benchmarks
### Naive version
run run_base.sh
### Single machine training with pipeline  
run run_single.sh, please write the --watch-gpu the same id as the visible gpu
### Distributed training  
run ps.sh  
run worker0.sh --watch-gpu=0  
run worker1.sh  --watch-gpu=1  
set watch-gpu id the same as visible gpu to each one. There could be a problem that some can always see first gpu.
It should have been solved by adding in the code ""

# Notice
## About the save steps of profiler
- Save steps of profiling could be 60, but not too small. Save summary rarely.
## Watch gpu may be left running
nvidia-smi and cpu log start during running, and will be killed with Ctrl+C keyboard interruption or the training end itself normally.  
If the training process end because of lack of memory. nvidia-smi has to be killed manually

## Part of benchmarks best parameter settings
Pacalvoc must be downloaded otherwise 0 feeded. 
Frequent saving profiler and summary take a lot of memory, better save profiler every 60s, save summary rarely, like 500s
queue_size seems doesnt need to be too big, 2 or 3 should be fine, because anyway each training iteration take some time 

# For myself report and reasons:
## Some explanations for distributed training bottleneck on 1G ethernet, 1 gpu on each machine
https://discuss.mxnet.io/t/multi-system-multi-gpu-distributed-training-slower-than-single-system-multi-gpu/1270
https://github.com/tensorflow/ecosystem/issues/69
https://github.com/tensorflow/tensorflow/issues/2397 most detailed.  

byronyi commented on 20 Nov 2017
Get a proper network if you plan to do distributed training of large models.
Bottom line: don't do distributed TF with <10Gbps network...

## My explanatoins
- batch set to very small ,like 2. two machines work. so the code works. gpu of ps and worker0 is 40-80%, some space ot increase batch size
- But even with this small batch size, there is big speed gap between two machines. worker1 that ps doesnt reside on passed 4 iterations while worker0 passed 100 iterations
- Reasons from me:
  gRPC based on protobuf is not ideal protocal for distributed deep learning, it is not dedicated designed for it anyway  
  CPU receive the model parameters, and then memcopy, take time and space.  
  ethernet speed less than 10G, not enough from someone's commenets  

