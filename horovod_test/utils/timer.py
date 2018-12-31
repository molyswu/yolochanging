import time
import datetime
import yolo.config as cfg
class Timer(object):
    def __init__(self):
        self.global_started = False;
        self.global_start_time = 0
        self.start_global_step_value = 0
        self.global_diff = 0.
        self.remain_time = 0.

    def tic(self, global_restart=False, start_global_step_value=0):
        if global_restart == True:
            self.global_started = True
            self.global_start_time = time.time()
            self.start_global_step_value = start_global_step_value
        self.tick_start_time = time.time()

    def toc(self, iters_per_toc, global_step_value):
        self.diff = time.time() - self.tick_start_time
        local_iter_fps = float(cfg.BATCH_SIZE/self.diff)
        global_avg_step = 0
        if self.global_started == True:
            self.global_diff = time.time() - self.global_start_time
            global_avg_fps = (global_step_value - self.start_global_step_value)*cfg.BATCH_SIZE/self.global_diff 
        return local_iter_fps, global_avg_fps

    def remain(self, current_global_step, stop_global_step):
        if self.global_started == True:
            self.remain_time = self.global_diff / (current_global_step - self.start_global_step_value)* (stop_global_step - current_global_step)
            return str(datetime.timedelta(seconds=int(self.remain_time)))
