import tensorflow as tf
import yolo.dis_config as cfg


ps_hosts = cfg.PS_HOSTS.split(",")
worker_hosts = cfg.WORKER_HOSTS.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster,
                         job_name='ps',
                         task_index=0)
    
server.join()
