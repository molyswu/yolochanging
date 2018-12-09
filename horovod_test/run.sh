mpirun -np 4 \
	-H localhost:4 \
	-bind-to none -oversubscribe \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x $LD_LIBRARY_PATH \
	-mca pml ob1 -mca btl ^openib \
	\
	python3 horovod_distributed_mpi.py \
