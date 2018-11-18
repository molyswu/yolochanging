mport os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net_naive import YOLONet
from utils.pascal_voc_naive import pascal_voc
from utils.timer import Timer

slim = tf.contrib.slim

#TODO checkpoints dont work
#TODO inputpipeline for inference
#TODO make a version with inputpipe line but training locally.


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    # parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    FLAGS, unparsed = parser.parse_known_args()
    
    # if FLAGS.data_dir != cfg.DATA_PATH:
    #     update_config_paths(FLAGS.data_dir, FLAGS.weights)
    
    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    print('Start training ...')
    
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
    
    tensors_to_log = [global_step, yolo.total_loss]

    checkpoint_dir = "checkpoints"
    summary_dir="summary_dir"
    profiler_dir="profiler_dir"
    def formatter(curvals):
        print("Global step %d, GenLoss %f!" % (
            curvals[global_step], curvals[yolo.total_loss]))
    
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50, formatter=formatter)
    profiler_hook = tf.train.ProfilerHook(save_secs=60, output_dir=profiler_dir, show_memory=True)
    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_secs=30, output_dir=summary_dir, summary_op=summary_op)

    checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir, save_steps=50, checkpoint_basename = 'YOLO_small.ckpt')

    hooks = [tf.train.StopAtStepHook(last_step = 2000), logging_hook, profiler_hook, summary_hook, checkpoint_hook]
    #last step is global step
    
    with tf.train.MonitoredSession(hooks=hooks) as sess:
        
        timer = Timer()
        local_iter = 0
        
        while not sess.should_stop():
            local_iter = local_iter + 1
            timer.tic()
            images, labels = pascal.get_batch()
            feed_dict = {yolo.images: images, yolo.labels: labels}
            current_iter_load_time = timer.tocInput()

            timer.tic()
            yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op], feed_dict=feed_dict)
            (diff_current_iter, current_iter_fps, local_average_fps) = timer.toc(local_iter, global_step_value)
            print("yolo_total_loss %f, global_step %d" % (yolo_loss, global_step_value))
            if local_iter % 10 == 0 and local_iter > 0:
                log_str = 'Local iter %d, batch_size %d, current iter load time: %fs, current process time: %fs current_iter_fps input excluded %f, local_average_fps: %f' % (local_iter, cfg.BATCH_SIZE, current_iter_load_time, diff_current_iter, current_iter_fps, local_average_fps)
                #Local iter 10, batch_size 40, current iter load time: 0.927860s, current process time: 48.237226s current_iter_fps input excluded 0.829235, local_average_fps: 0.816957,
                #Local iter 10, batch_size 24, current iter load time: 0.453948s, current process time: 28.765632s current_iter_fps input excluded 0.834329, local_average_fps: 0.816195
                #Local iter 10, batch_size 24, current iter load time: 0.371289s, current process time: 28.804506s current_iter_fps input excluded 0.833203, local_average_fps: 0.815096
                print(log_str)


if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
