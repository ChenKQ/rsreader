import cv2
import numpy as np
from .PILReader import PILReader

class OpenCVReader(PILReader):
    '''
    A low-level tool based on OpenCV to read an image which is specified by the parameter "imgfile".
    '''
    def __init__(self,imgfile):
        '''
        Constructor. It reads the image from the disk into the memory \
        and stores the contents in the format of numpy.ndarray with the shape of (height, width, channel). \
        Since opencv reads images in the orders of bgr instead of rgb which is adopted by most library.
        :param imgfile: the image file to read
        '''
        self.imgfile = imgfile
        self.img = cv2.imread(self.imgfile, -1) # (height,width,channel) in bgr order
        if len(self.img.shape) ==2:
            self.img = np.expand_dims(self.img,2)
        else:
            self.img[...,:] = self.img[...,(2,1,0)] ## change the order of channels to rgb