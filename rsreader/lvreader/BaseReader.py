import numpy as np

class BaseReader(object):
    '''
    This is a low-level reader to read images which does not implement the details. \
    The implementation is done by GdalReader, OpenCVReader, PILReader or other readers.
    '''
    def __init__(self,imgfile,readertype='pil'):
        '''
        Constructor.
        :param imgfile: The image file to read.
        :param readertype: which type of reader is used to read the image. If readertype is 'pil', \
                           the PILReader will be used. If readertype is 'gdal', the GdalReader will be used.\
                           If readertype is 'opencv', the OpenCVReader will be used. \
                           If it is defined in other places, please use the class name without citation \
                           where the corresponding interfaces should be implemented. \
                           It cannot be None. In default, the PILReader will be used.
        '''
        self.imgfile = imgfile
        if readertype =='gdal':
            from .GdalReader import GdalReader
            self.reader = GdalReader(self.imgfile)
        elif readertype == 'opencv':
            from .OpenCVReader import OpenCVReader
            self.reader = OpenCVReader(self.imgfile)
        elif readertype =='pil':
            from .PILReader import PILReader
            self.reader = PILReader(self.imgfile)
        elif readertype is None:
            assert 'readertype should not be None'
        else:
            self.reader = readertype(self.imgfile)

    def readPatch(self,startx,starty,width,height,bandlst=[],dtype=np.uint8):
        '''
        Crop a patch from the complete image. The left-upper point is the starting point.
        :param startx: The x-dimension of the left-upper point of the crop in the original image.
        :param starty: The y-dimension of the left-upper point of the crop in the original image.
        :param width: The width of the crop. If the parameter "width" is smaller than 0, \
                      the crop covers the whole range of the width of the image.
        :param height: The height of the crop. If the parameter "height" is smaller than 0, \
                       the crop covers the whole range of the height of the image.
        :param bandlst: the channels (bands) to be read.
        :param dtype: the data type used to store the image. It can be numpy.uint8, np.uint16 and so on.
        :return: the crop from the original image. It is stored as numpy.ndarray with the shape of (channel, height, width)
        '''
        return self.reader.readPatch(startx,starty,width,height,bandlst,dtype)

    def readImg(self,bandlst=[],dtype=np.uint8):
        '''
        Read the complete image.
        :param bandlst: the channels (bands) to be read
        :param dtype: the data type used to store the image. It can be numpy.uint8, np.uint16 and so on.
        :return: the image read into the memory in the format of numpy.ndarray with the shape of (channel, heigh, width)
        '''
        return self.reader.readImg(bandlst,dtype)

    def getNChannel(self):
        '''
        Get the number of channels of the image.
        :return: (int) the number of channels of the image
        '''
        return self.reader.getNChannel()

    def getSize(self):
        '''
        Get the size of the image.
        :return: the size of the image with the format of a tuple (height, width)
        '''
        return self.reader.getSize()

