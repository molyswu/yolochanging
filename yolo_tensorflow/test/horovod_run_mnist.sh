export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/macierz/home/s170011/miniconda3/envs/py36tf/lib
export PATH=$PATH:/macierz/home/s170011/miniconda3/envs/py36tf/lib
mpirun -np 1 \
    -H des10.kask.eti.pg.gda.pl:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -mca pml ob1 -mca btl ^openib \
    python horovod_tensorflow_mnist.py
