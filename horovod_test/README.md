#Fuck this folder in master doesnt work. To check horovod scalability, checkout horovod-test branch

# In this horovod\_test folder, only the file distributed\_mpi.py and run\_distributed\_mpi.sh is used. The single etc, are not used and also not checked.

# The reason is horovod make the single node training and distributed training no difference, because horovod is based on mpi, not like tensorflow based on TCP and gRPC protocal. So we only need to run one bash on one machine. Everything is much faster.
