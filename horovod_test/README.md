# In this horovod\_test folder, only the file distributed\_mpi.py and run\_distributed\_mpi.sh is used. nopm\_distributed_mpi.py The single etc, are not used and also not checked.

# The reason is horovod make the single node training and distributed training no difference, because horovod is based on mpi, not like tensorflow based on TCP and gRPC protocal. So we only need to run one bash on one machine. Everything is much faster.


# The running instructions:
switch on the virtual instance   
gcloud compute ssh danqing@vm-us-west1-a-cpu20--ram75g-gpu8-tesla-v100
