mpirun -np 1 \
    -H des10.kask:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    python horovod_train_singlepipeline.py
