mpirun -np 4 \
	-H localhost:4 \
	-bind-to none -map-by slot \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x $LD_LIBRARY_PATH \
	-mca pml ob1 -mca btl ^openib \
	\
	python horovod_train_singlepipeline.py \
	--watch_gpu=0
