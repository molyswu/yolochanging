import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.pascal_voc import Pascal_voc
import tensorflow.contrib.slim as slim
from utils.timer import Timer
import subprocess
import time
import horovod.tensorflow as hvd

# NO GPU watch, because this time it will be 4 gpus
# stop_global will be read from the config.py
# No checkpoints, set the checkpoints very big 
# No task_index
# mpi, all processes one piece of code

def main():
    if os.path.exists(cfg.LOG_DIR):
        os.system("rm -rf %s"%cfg.LOG_DIR)  
    tf.logging.set_verbosity(tf.logging.INFO)
    
    hvd.init()

    log_dir = cfg.LOG_DIR if hvd.rank()==0 else None  

    stop_global_step = cfg.STOP_GLOBAL_STEP   
    prof_save_step = cfg.PROFILER_SAVE_STEP #120
    sum_save_step = cfg.SUMMARY_SAVE_STEP #500
    checkpoint_save_step = cfg.CHECKPOINT_SAVE_STEP
    
    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    
    ########################dir#######################################
    
    #########################pipeline###################################
    tf.reset_default_graph()
    
    image_producer = Pascal_voc('train')
    
    (image, label) = image_producer.get_one_image_label_element()

    image_shape = (image_producer.image_size, image_producer.image_size, 3)

    label_size = (image_producer.cell_size, image_producer.cell_size, 25)  # possible value is 0 or 1

    processed_queue = tf.FIFOQueue(capacity=int(image_producer.batch_size * cfg.MUL_QUEUE_BATCH), shapes = [image_shape, label_size], dtypes = [tf.float32, tf.float32], name = 'processed_queue')

    enqueue_processed_op = processed_queue.enqueue([image, label])

    num_enqueue_threads = min(image_producer.num_enqueue_threads, image_producer.gt_labels_length)

    queue_runner = tf.train.QueueRunner(processed_queue, [enqueue_processed_op] * num_enqueue_threads)
    tf.train.add_queue_runner(queue_runner)

    (images, labels) = processed_queue.dequeue_many(image_producer.batch_size)
    #if FLAGS.debug == True:
    #    labels = tf.Print(labels, data=[processed_queue.size()],
    #                      message="Worker %d get_batch(), BatchSize %d, Queues left:" % (
    #                          FLAGS.task_index, cfg.BATCH_SIZE))

    #########################graph###################################
   
    #print("Current rank is %d"%hvd.rank())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list=str(hvd.local_rank())
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
 
    yolo = YOLONet(images, labels)
        
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
    initial_learning_rate, global_step, decay_steps,
    decay_rate, staircase, name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
   
    #optimizer=tf.train.AdagradOptimizer(0.01*hvd.size())
    optimizer=hvd.DistributedOptimizer(optimizer) 
    train_op = slim.learning.create_train_op(yolo.total_loss, optimizer, global_step=global_step)
        
    #########################hook#####################################
    
    #profiler_hook = tf.train.ProfilerHook(save_steps=prof_save_step, output_dir=log_dir, show_memory=True,show_dataflow=True)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=sum_save_step, output_dir=log_dir, summary_op=summary_op)
   
   # tf.train.LoggingTensorHook(tensors={'step':global_steo,'loss':yolo.total_loss},every_n_iter=10) 
    tensors_to_log = [global_step, yolo.total_loss]
    def formatter(curvals):
        print("Global step %d, Loss %f!" % (
            curvals[global_step], curvals[yolo.total_loss]))
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10, formatter=formatter)
   
    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
    # from rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights
# or restored from a checkpoint.
    pm=PerformanceHook(50,)
    hooks = [hvd.BroadcastGlobalVariablesHook(0), tf.train.StopAtStepHook(last_step=stop_global_step), pm]
    #######################train#####################################
    
    print('Start training ...')
    with tf.train.MonitoredTrainingSession(config=config, hooks=hooks, checkpoint_dir=log_dir) as sess:
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op])
        while not sess.should_stop():
            yolo_loss, gs_step, _ = sess.run([yolo.total_loss, global_step, train_op])
        
        coord.request_stop()
        coord.join(threads)
        
    print('Done training.')

class PerformanceMeasureHook(tf.train.SessionRunHook):
    def __init__(self, every_n_step, global_step):
        self.last_time = time.time()
        self.last_gs = 0
        self.global_step = global_step
        self.every_n_step = every_n_step
    
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.global_step)
    
    def after_run(self, run_context, run_values):
        cur_gs = run_values.results
        cur_time = time.time()
        if cur_gs - self.last_gs > self.every_n_step:
   
            # the thing is if we have stable speed, we dont need global average. but for two tcp, it is needed, because there are difference. and in horovod nccl, it is synchronous gradient descent, so anyway slower one have to wait for faster one   
            speed = (cur_gs - self.last_gs) * batch_size / (cur_time - self.last_time)
            
            if(self.last_gs < batch_size*3):
                 print('Iteration: %d, Warm-up, global_step: %d, Speed: %d fps'%(cur_gs/batch_size, cur_gs, speed))
            else:
                 print('Iteration: %d, global_step: %d, Speed: %d fps'%(cur_gs/batch_size, cur_gs, speed))
            self.last_gs = cur_gs 
            self.last_time = cur_time
        
if __name__ == '__main__':
    main()
