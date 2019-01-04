mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    python horovod_train_singlepipeline.py
