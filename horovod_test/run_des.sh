mpirun -np 2 \
    -H  localhost:1,172.20.83.217:1\
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python3 nopmhook_horovod_distributed_mpi.py
