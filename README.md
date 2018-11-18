# yolochanging


#Naive version:
run base_run.sh

#single machine training
run single_run.sh, please write the --watch-gpu the same as the visible gpu

#distributed training with pipeline
run ps.sh
run worker0.sh --watch-gpu=0
run worker1.sh  --watch-gpu=1
set different visible gpu to each one. There could be a problem that some can always see first gpu.
It should be solved by adding in the code

increase the batch size and queue size. watch the profiler json so that gpu is fully used and cpu do not be idle for too long time
