mpirun -np 2 \
    -H  localhost:2\
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python3 horovod_distributed_mpi.py
