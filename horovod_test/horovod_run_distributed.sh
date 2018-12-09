mpirun -np 4 \
    -H  localhost:4\
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python horovod_distributed_mpi.py
