import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net_naive import YOLONet
from utils.pascal_voc_naive import pascal_voc
from utils.timer import Timer
import os
import subprocess

slim = tf.contrib.slim

#TODO checkpoints dont work
#TODO inputpipeline for inference
#TODO make a version with inputpipe line but training locally.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch_gpu', required=True, type=int,
                        help="watch gpu id filled Set it the same as visible gpu id")
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--stop_globalstep', default=1000, type=int)
    parser.add_argument('--checkpoint_dir', default="checkpoint_dir", type=str)
    parser.add_argument('--task_index', default=0, type=int)
    
    FLAGS, unparsed = parser.parse_known_args()

    prof_save_step = cfg.PROFILER_SAVE_STEP  # 120
    sum_save_step = cfg.SUMMARY_SAVE_STEP  # 500
    FLAGS, unparsed = parser.parse_known_args()

    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    
    
    ########################dir#######################################
    singlepipe_dir = "Single_Naive_train_logs"
    if not os.path.exists(singlepipe_dir):
        os.makedirs(singlepipe_dir)

    inside_bsnQnM_dir = "Single_Naive" + cfg.BS_NT_MUL_PREFIX
    logrootpath = os.path.join(singlepipe_dir, inside_bsnQnM_dir)
    if not os.path.exists(logrootpath):
        os.makedirs(logrootpath)

    fpslog_name = "Single_Naive" + cfg.BS_NT_MUL_PREFIX + "fps_log.txt"
    concated_path = logrootpath + "/" + fpslog_name

    checkpoint_dir = FLAGS.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    gpulog_name = "Single_Naive" + "gpu" + str(FLAGS.watch_gpu) + cfg.BS_NT_MUL_PREFIX + "_gpulog.txt"

    ###########################gpusubprocess##############################
    def start_gpulog(path, fname):
        # has to be called before start of training
        gpuinfo_path = path + "/" + fname
        with open(gpuinfo_path, 'w'):
            argument = 'timestamp,count,gpu_name,gpu_bus_id,memory.total,memory.used,utilization.gpu,utilization.memory'
        try:
            proc = subprocess.Popen(
                ['nvidia-smi --format=csv --query-gpu=%s %s %s %s' % (
                argument, ' -l 1', '-i ' + str(FLAGS.watch_gpu), '-f ' + gpuinfo_path)], shell=True)
        except KeyboardInterrupt:
            try:
                proc.terminate()
            except OSError:
                pass
                proc.kill()
        return proc

    tf.reset_default_graph()

    with tf.device("/device:GPU:" + str(FLAGS.watch_gpu)):
        yolo = YOLONet()
    
        pascal = pascal_voc('train')
        
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, decay_steps,
            decay_rate, staircase, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(
            yolo.total_loss, optimizer, global_step=global_step)
   

    profiler_hook = tf.train.ProfilerHook(save_steps=prof_save_step, output_dir=logrootpath, show_memory=True,
                                          show_dataflow=True)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=sum_save_step, output_dir=logrootpath, summary_op=summary_op)

    if FLAGS.debug == True:
        tensors_to_log = [global_step, yolo.total_loss]
    
        def formatter(curvals):
            print("Global step %d, Loss %f!" % (
                curvals[global_step], curvals[yolo.total_loss]))
    
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10, formatter=formatter)
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.stop_globalstep), logging_hook, profiler_hook, summary_hook]
    else:
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.stop_globalstep), profiler_hook, summary_hook]

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    # config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.8   #do not assign all at the begining
    proc = start_gpulog(logrootpath, gpulog_name)
    
    #Forgot to put checkpoint, but ehh anyway not use it here
    
    
    with tf.train.MonitoredTrainingSession(hooks=hooks,config=config) as sess:
        start_global_step_value = sess.run(global_step)
        timer = Timer(start_global_step_value)
        # local_iter = 0
        
        iters_per_toc = 20
        txtForm = "Training speed: local avg %f fps, global %f fps, loss %f, global step: %d, predict to wait %s"
        local_max_iter = FLAGS.stop_globalstep - start_global_step_value

        timer.tic()
        images_data, labels_data = pascal.get_batch()
        feed_dict = {yolo.images: images_data, yolo.labels: labels_data}
        yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op], feed_dict=feed_dict)
        n = 0
        while not sess.should_stop():
            n = n + 1
            if n > 0 and n % iters_per_toc == 0:
                if n > 0 and n % iters_per_toc == 0:
                    local_avg_fps, global_avg_fps = timer.toc(iters_per_toc, global_step_value)
                    timetowait = timer.remain(n, local_max_iter)
            
                    txtData = local_avg_fps, global_avg_fps, yolo_loss, global_step_value, timetowait
                    print(txtForm % txtData)
            
                    with open(concated_path, 'a+') as log:
                        log.write("%.4f,%.4f,%.4f,%d,%s\n" % txtData)
            
                    timer.tic()
            #yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op])
            images_data, labels_data = pascal.get_batch()
            feed_dict = {yolo.images: images_data, yolo.labels: labels_data}
            yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op], feed_dict = feed_dict)
        
        print('Done training.')
        
        try:
            proc.terminate()
        except OSError:
            pass
            print("Kill subprocess failed. Kill nvidia-smi mannually")
            # local_iter = local_iter + 1
            # timer.tic()
            # images, labels = pascal.get_batch()
            # feed_dict = {yolo.images: images, yolo.labels: labels}
            # current_iter_load_time = timer.tocInput()
            #
            # timer.tic()
            # yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op], feed_dict=feed_dict)
            # (diff_current_iter, current_iter_fps, local_average_fps) = timer.toc(local_iter, global_step_value)
            # print("yolo_total_loss %f, global_step %d" % (yolo_loss, global_step_value))
            # if local_iter % 10 == 0 and local_iter > 0:
            #     log_str = 'Local iter %d, batch_size %d, current iter load time: %fs, current process time: %fs current_iter_fps input excluded %f, local_average_fps: %f' % (local_iter, cfg.BATCH_SIZE, current_iter_load_time, diff_current_iter, current_iter_fps, local_average_fps)
                #Local iter 10, batch_size 40, current iter load time: 0.927860s, current process time: 48.237226s current_iter_fps input excluded 0.829235, local_average_fps: 0.816957,
                #Local iter 10, batch_size 24, current iter load time: 0.453948s, current process time: 28.765632s current_iter_fps input excluded 0.834329, local_average_fps: 0.816195
                #Local iter 10, batch_size 24, current iter load time: 0.371289s, current process time: 28.804506s current_iter_fps input excluded 0.833203, local_average_fps: 0.815096
                # print(log_str)

    # tensors_to_log = [global_step, yolo.total_loss]
    #
    # checkpoint_dir = "checkpoints"
    # summary_dir="summary_dir"
    # profiler_dir="profiler_dir"
    # def formatter(curvals):
    #     print("Global step %d, GenLoss %f!" % (
    #         curvals[global_step], curvals[yolo.total_loss]))
    #
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50, formatter=formatter)
    # profiler_hook = tf.train.ProfilerHook(save_secs=60, output_dir=profiler_dir, show_memory=True)
    # summary_op = tf.summary.merge_all()
    # summary_hook = tf.train.SummarySaverHook(save_secs=30, output_dir=summary_dir, summary_op=summary_op)
    #
    # checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir, save_steps=5000, checkpoint_basename = 'YOLO_small.ckpt')
    #
    # hooks = [tf.train.StopAtStepHook(last_step = FLAGS.stop_globalstep), logging_hook, profiler_hook, summary_hook, checkpoint_hook]
    # last step is global step

    #########################hook#####################################

if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
