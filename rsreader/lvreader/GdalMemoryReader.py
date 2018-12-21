import numpy as np
from .GdalReader import GdalReader

class GdalMemoryReader(GdalReader):
    '''
    A low-level tool based on GDAL to read an image which is specified by the parameter "imgfile". \
    It can read common GEO-TIFF and img files which cannot be read by OpenCV or PIL. \
    Compared with the GdalReader, this class will load the image into memories firstly so as to accelerate the speed. \
    It can be used to the situation when the image is not too large to load into the memory.
    '''
    def __init__(self,imgfile):
        '''
        Constructor. It opens the image and load the basic information about the image like size, channels, and so on. \
        It stores the image into the memory in the format of numpy.ndarray with the shape of (channel, height, width).
        :param imgfile: the image file to read.
        '''
        self.imgfile=imgfile
        self.bands=[]
        self._open()
        self._getBands_and_read_image_into_memory()

    def _getBands_and_read_image_into_memory(self):
        '''
        This method gets all bands of tiff file and these bands are stored in self.bands. It is a list type.
        Then the contents of the image will be loaded into memory as the numpy data type.
        :return: None
        '''
        assert not self.dataset is None
        self.img = []
        for i in range(self.dataset.RasterCount):
            band=self.dataset.GetRasterBand(i+1)
            self.bands.append(band)
            self.img.append(band.ReadAsArray())
        self.img = np.asarray(self.img)

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
        for idx in bandlst:
            ret[idx-1,sy:ey,sx:ex] = img[idx-1, fsy:fey, fsx:fex]
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
        ret = self.img[np.asarray(bandlst)-1,...]
        ret = ret.astype(dtype=dtype)
        return ret
