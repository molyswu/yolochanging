import time
import datetime
import yolo.config as cfg

class Timer(object):
    def __init__(self, start_global_step_value):
        self.global_start_time = time.time()
        self.tick_start_time = 0
        self.total_time = 0.
        self.start_global_step_value = start_global_step_value
        self.local_diff = 0.
        self.global_diff = 0
        self.remain_time = 0.

    def tic(self):
        self.tick_start_time = time.time()

    def toc(self, iters_per_toc, global_step_value):
        self.diff = time.time() - self.tick_start_time
        self.global_diff = time.time() - self.global_start_time
        self.local_iter_fps = float(iters_per_toc*cfg.BATCH_SIZE/self.diff)
        self.global_average_fps = float((global_step_value-self.start_global_step_value)*cfg.BATCH_SIZE/self.global_diff)
        return self.local_iter_fps, self.global_average_fps

    def remain(self, iters, max_iters):
        self.remain_time = (time.time() - self.global_start_time) / iters *  (max_iters - iters)
        return str(datetime.timedelta(seconds=int(self.remain_time)))
