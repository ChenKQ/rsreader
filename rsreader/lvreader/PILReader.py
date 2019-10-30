from PIL import Image
import numpy as np


class PILReader(object):
    '''
    A low-level tool based on PIL.Image to read an image which is specified by the parameter "imgfile".
    '''
    def __init__(self,imgfile):
        '''
        Constructor. It reads the image from the disk into the memory \
        and stores the contents in the format of numpy.ndarray with the shape of (height, width, channel).
        :param imgfile: the image file to read.
        '''
        self.imgfile = imgfile
        self.pil = Image.open(self.imgfile)
        self.img = np.asarray(self.pil)
        if len(self.img.shape) ==2:
            self.img = np.expand_dims(self.img,2)

    def getNChannel(self):
        '''
        Get the number of channels of the image.
        :return: (int) the number of channels of the image
        '''
        return self.img.shape[2]

    def getSize(self):
        '''
        Get the size of the image.
        :return: the size of the image with the format of a tuple (height, width)
        '''
        return self.img.shape[0:2]

    def readPatch(self,startx,starty,width,height,bandlst=[],dtype=np.uint8):
        '''
        Crop a patch from the complete image. The left-upper point is the starting point.
        :param startx: The x-dimension of the left-upper point of the crop in the original image.
        :param starty: The y-dimension of the left-upper point of the crop in the original image.
        :param width: The width of the crop. If the parameter "width" is smaller than 0, \
                      the crop covers the whole range of the width of the image.
        :param height: The height of the crop. If the parameter "height" is smaller than 0, \
                       the crop covers the whole range of the height of the image.
        :param bandlst: The channels (bands) to be read. If is is 'None' or the length is 0,
                        all bands will be read. It starts from 1 instead of 0.
        :param dtype: the data type used to store the image. It can be numpy.uint8, np.uint16 and so on.
        :return: the crop from the original image. It is stored as numpy.ndarray with the shape of (channel, height, width)
        '''
        img = self.img
        imgsize = self.getSize()
        if width<0:
            width = imgsize[1]
        if height<0:
            height = imgsize[0]
        if bandlst is None or len(bandlst)==0:
            bandlst = range(1,self.getNChannel()+1)
        sy = -min(0, starty)
        ey = min(height, imgsize[0] - starty)
        sx = -min(0, startx)
        ex = min(width, imgsize[1] - startx)
        fsy = max(0, starty)
        fey = max(0, starty) + min(height, height + starty, imgsize[0] - starty)
        fsx = max(0, startx)
        fex = max(0, startx) + min(width, width + startx, imgsize[1] - startx)
        ret = np.zeros((len(bandlst), height, width), dtype=dtype)
        for dst_idx, idx in enumerate(bandlst):
            ret[dst_idx,sy:ey,sx:ex] = img[fsy:fey,fsx:fex, idx-1]
        return ret

    def readImg(self,bandlst=[],dtype=np.uint8):
        '''
        Read the complete image.
        :param bandlst: the channels (bands) to be read
        :param dtype: the data type used to store the image. It can be numpy.uint8, np.uint16 and so on.
        :return: the image read into the memory in the format of numpy.ndarray with the shape of (channel, heigh, width)
        '''
        if bandlst is None or len(bandlst)==0:
            bandlst = range(1,self.getNChannel()+1)
        ret = self.img[...,np.asarray(bandlst)-1] # h,w,c
        ret = np.transpose(ret, (2,0,1)) # c, h, w
        ret = np.asarray(ret, dtype=dtype)
        return ret
        # return self.readPatch(0, 0, -1, -1, bandlst, dtype)

