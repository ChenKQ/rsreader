import os, time
import numpy as np

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdirs(dir_name)

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

def linearStrech(img, low_p=2, high_p=98):
    '''
    Image should be in shape of (W,C)
    '''
    img = img.astype(np.int64)
    lowest_2p = np.percentile(img,low_p)
    highest_2p = np.percentile(img,high_p)
    img_cut = (img<=highest_2p)*(img>=lowest_2p)*img
    img_cut += (img>highest_2p)*int(highest_2p)
    img_cut += (img<lowest_2p)*int(lowest_2p)
    img = img_cut
    img = (img-lowest_2p)/(highest_2p-lowest_2p)
    img *= 255
    return img