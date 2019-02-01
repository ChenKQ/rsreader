import numpy as np
import cv2
import numbers
import random

class DataAugument(object):
    def __init__(self,funcs=[]):
        self.funcs = funcs

    def getScale(self):
        sx = 1.0
        sy = 1.0
        for f in self.funcs:
            scale = f.getscale()
            sy *= scale[0]
            sx *= scale[1]
        return (sy,sx)

    def aug(self,data,label):
        for f in self.funcs:
            data,label = f(data,label)
        return data,label

    def recoverScore(self,lb):
        for f in reversed(self.funcs):
            lb = f.recoverScore(lb)
        return lb

def randomBool():
    xx=[True,False]
    return random.sample(xx,1)[0]

class Rotate(object):
    def __init__(self,patchsize,seed=None):
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize,patchsize)
        else:
            self.patchsize = patchsize
        self.seed=seed

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        xx = [0, 1, 2, 3]
        if self.seed is None:
            rotateAngle = random.sample(xx, 1)[0]
        elif self.seed<0:
            rotateAngle = xx[0]
        else:
            rotateAngle = xx[self.seed]
        for i in range(img.shape[0]):
            img[i, ...] = np.rot90(img[i, ...], rotateAngle)
        label = np.rot90(label, rotateAngle)
        return img, label

    def recoverScore(self,lb):
        if self.seed is None:
            pass
        elif self.seed<0:
            pass
        else:
            ang = (4-self.seed)%4
            # print 'rotate angle:', ang
            for i in range(lb.shape[0]):
                lb[i, ...] = np.rot90(lb[i, ...], ang)
        return lb

    def getscale(self):
        return (1,1)


class FreeRotate(object):
    def __init__(self,patchsize, seed=None, step=1):
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize,patchsize)
        else:
            self.patchsize = patchsize
        self.seed=seed
        self.choices = range(0,360,step)

    @staticmethod
    def rotate(img, angle, chw=True, flags = cv2.INTER_LINEAR):
        if len(img.shape) == 2:
            rows, cols = img.shape[0:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            dst = cv2.warpAffine(img[...], M, (cols, rows), flags = flags,borderValue=255)
            return dst
        if chw:
            for i in range(img.shape[0]):
                rows, cols = img.shape[1:3]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                dst = cv2.warpAffine(img[i], M, (cols, rows), flags = flags)
                img[i, :] = dst
        else:
            for i in range(img.shape[2]):
                rows, cols = img.shape[0:2]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                dst = cv2.warpAffine(img[..., i], M, (cols, rows), flags = flags)
                img[..., i] = dst
        return img

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        if self.seed is None:
            rotateAngle = random.sample(self.choices, 1)[0]
        elif self.seed<0:
            rotateAngle = 0
        else:
            rotateAngle = self.seed
        img = FreeRotate.rotate(img, rotateAngle,True)
        label = FreeRotate.rotate(label, rotateAngle, False, flags=cv2.INTER_NEAREST)
        return img, label

    def getscale(self):
        return (1,1)


class Mirror(object):
    def __init__(self,patchsize,seed=None):
        '''

        :param patchsize:
        :param seed: None for random, <0 for forbid, >0 for force
        '''
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize,patchsize)
        else:
            self.patchsize = patchsize
        self.seed = seed

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        if self.seed is None:
            isHorizonFlip = randomBool()
        elif self.seed <0:
            isHorizonFlip = False
        else:
            isHorizonFlip = True
        if isHorizonFlip:
            img = img[:, ::-1, :]
            label = label[::-1, :]
        return img, label

    def recoverScore(self,lb):
        if self.seed is None:
            pass
        elif self.seed <0:
            pass
        else:
            lb = lb[:,::-1,:]
        return lb

    def getscale(self):
        return (1,1)

class InjectGaussianNoise(object):
    def __init__(self,patchsize,mean=0.0,std=1.0,seed=None):
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize,patchsize)
        else:
            self.patchsize = patchsize
        self.mean=mean
        self.std = std
        self.seed = seed

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        if self.seed is None or self.seed >0:
            noise = np.random.normal(self.mean, self.std, img.shape)
            img += noise
        elif self.seed <0:
            pass
        return img, label

    def recoverScore(self,lb):
        return lb

    def getscale(self):
        return (1,1)

class ScaleAug(object):
    def __init__(self,patchsize,symmetry=True,scale_range=[0.25,2.0],eps=1000,seed=None):
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize,patchsize)
        else:
            self.patchsize = patchsize
        self.symmetry = symmetry
        self.scale_range = scale_range
        self.eps = eps
        self.seed = seed
        self.__getSet()

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        ret = np.zeros((img.shape[0],self.patchsize[0],self.patchsize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            ret[i] = cv2.resize(img[i],dsize=(self.patchsize[1],self.patchsize[0]))
        label = cv2.resize(label,dsize=(self.patchsize[1],self.patchsize[0]),interpolation=cv2.INTER_NEAREST)
        return ret,label

    def recoverScore(self,lb):
        return lb

    def __getSet(self):
        self.allset = np.arange(self.scale_range[0]*self.eps,self.scale_range[1]*self.eps)
        self.allset = self.allset.astype(np.float32)/self.eps
        self.allset = list(self.allset)

    def getscale(self):
        if self.seed is None:
            s1 = random.sample(self.allset,1)[0]
            if self.symmetry:
                s2 = s1
            else:
                s2 = random.sample(self.allset,1)[0]
            self.scale = (s1,s2)
        elif self.seed <0:
            self.scale = (1,1)
        else:
            self.scale = self.seed
        return self.scale

class RandomCrop(object):
    def __init__(self, scale_range, dSize,symmetry=True, eps = 1000):
        self.scale_range = scale_range
        if isinstance(dSize, numbers.Number):
            self.dSize = (int(dSize), int(dSize))
        else:
            self.dSize = dSize
        self.symmetry = symmetry
        self.eps = eps
        self.__getSet()

    def __getSet(self):
        self.allset = np.arange(self.scale_range[0]*self.eps,self.scale_range[1]*self.eps)
        self.allset = self.allset.astype(np.float32)/self.eps
        self.allset = list(self.allset)

    def __getCropSize(self, inSize):
        s1 = random.sample(self.allset,1)[0]
        if self.symmetry:
            s2 = s1
        else:
            s2 = random.sample(self.allset,1)[0]
        cropSize = (int(inSize[0]*s1), int(inSize[1]*s2))
        return cropSize

    def getscale(self):
        return (1, 1)

    @staticmethod
    def __randomCrop(patchsize, cropSize):
        return (int(round(np.random.rand()*(patchsize[0]-cropSize[0]))),
                int(round(np.random.rand()*(patchsize[1]-cropSize[1]))))

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        cropSize = self.__getCropSize(img.shape[1:])
        inity, initx = RandomCrop.__randomCrop(img.shape[1:], cropSize)
        ret = np.zeros((img.shape[0], self.dSize[0], self.dSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            cropped = img[i, inity:inity+cropSize[0], initx:initx+cropSize[1]]
            ret[i] = cv2.resize(cropped,dsize=(self.dSize[1],self.dSize[0]))
        cropped_label = label[inity:inity+cropSize[0], initx:initx+cropSize[1]]
        label = cv2.resize(cropped_label,dsize=(self.dSize[1],self.dSize[0]),interpolation=cv2.INTER_NEAREST)
        return ret,label

class SwapChannels(object):
    def __init__(self,patchsize,seed=None):
        '''

        :param patchsize:
        :param seed: None for random, <0 for forbid, >0 for force
        '''
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize,patchsize)
        else:
            self.patchsize = patchsize
        self.seed = seed

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        nchannel = img.shape[0]
        idx = list(range(0,nchannel))
        if self.seed is None:
            random.shuffle(idx)
        img[:,...] = img[idx,...]
        return img, label

    def recoverScore(self,lb):
        return lb

    def getscale(self):
        return (1,1)

