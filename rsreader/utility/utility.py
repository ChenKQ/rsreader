import os
import time

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

class GPUScheduler(object):
    def __init__(self, lock_dir='/home/chenkq/.gpulock',
                       lockfile='gpu.lock', sleeptime=0.01):
        self.lock_dir = lock_dir
        self.lockfile = lockfile
        self.gpu_lock = os.path.join(lock_dir,lockfile)
        self.sleeptime = sleeptime
        self.locked = False
        check_mkdir(self.lock_dir)

    def lock(self):
        print('waiting for the gpu\n')
        while(True):
            if os.path.exists(self.gpu_lock):
                time.sleep(self.sleeptime)
            else:
                os.mknod(self.gpu_lock)
                print('get the locker and lock sucessfully')
                self.locked = True
                return True

    def unlock(self):
        if os.path.exists(self.gpu_lock):
            os.remove(self.gpu_lock)
            print('unlock successfully')
            self.locked = False
        else:
            print('unlock')

    def __del__(self):
        print('release')
        if self.locked:
            self.unlock()