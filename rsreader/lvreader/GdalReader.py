from osgeo import gdal
from osgeo.gdalconst import *
import numpy as np

class GdalReader(object):
    '''
    A low-level tool based on GDAL to read an image which is specified by the parameter "imgfile". \
    It can read common GEO-TIFF and img files, which cannot be read by OpenCV or PIL. \
    Especially, it can be used in the cases of very large geo-images \
    when the whole image cannot be read into the memory at once.
    '''
    def __init__(self,imgfile):
        '''
        Constructor. It opens the image and load the basic information about the image like size, channels, and so on.
        :param imgfile: the image file to read.
        '''
        self.imgfile=imgfile
        self.bands=[]
        self._open()
        self._getBands()

    def _open(self):
        '''
        This method opens imgfile with gdal and it is stored in self.dataset
        :return: None
        '''
        self.dataset = gdal.Open(self.imgfile, GA_ReadOnly)

    def getTransform(self):
        return self.dataset.GetGeoTransform()

    def getProjection(self):
        return self.dataset.GetProjection()

    def getNChannel(self):
        '''
        Get the number of channels of the image.
        :return: (int) the number of channels of the image
        '''
        return self.dataset.RasterCount

    def getSize(self):
        '''
        Get the size of the image.
        :return: the size of the image with the format of a tuple (height, width)
        '''
        return (int(self.dataset.RasterYSize),int(self.dataset.RasterXSize))

    def _getBands(self):
        '''
        This method gets all bands of tiff file and these bands are stored in self.bands. It is a list type.
        :return: None
        '''
        assert not self.dataset is None
        for i in range(self.dataset.RasterCount):
            band=self.dataset.GetRasterBand(i+1)
            self.bands.append(band)

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
        if bandlst is None or len(bandlst)==0:
            bandlst = range(1,self.dataset.RasterCount+1)
        sy = -min(0, starty)
        ey = min(height, self.dataset.RasterYSize - starty)
        sx = -min(0, startx)
        ex = min(width, self.dataset.RasterXSize - startx)
        fsx = max(0, startx)
        fsy =  max(0, starty)
        flenthx = min(width, width + startx, self.dataset.RasterXSize - startx)
        fheighty =min(height, height + starty, self.dataset.RasterYSize - starty)
        img = np.zeros((len(bandlst), height, width), dtype=dtype)
        for idx in bandlst:
            img[idx - 1, sy:ey, sx:ex] = self.bands[idx-1].ReadAsArray(fsx,fsy,flenthx,fheighty)
        return img

    def readImg(self,bandlst=[],dtype=np.uint8):
        '''
        Read the complete image.
        :param bandlst: the channels (bands) to be read
        :param dtype: the data type used to store the image. It can be numpy.uint8, np.uint16 and so on.
        :return: the image read into the memory in the format of numpy.ndarray with the shape of (channel, heigh, width)
        '''
        if bandlst is None or len(bandlst)==0:
            bandlst = range(1,self.dataset.RasterCount+1)
        img = []
        for idx in bandlst:
            img.append(self.bands[idx-1].ReadAsArray())
        img = np.asarray(img, dtype=dtype)
        return img
        # return self.readPatch(0,0,self.dataset.RasterXSize,self.dataset.RasterYSize,bandlst,dtype)

    @classmethod
    def write(cls,outputPath,nbands,proj,trans,dataNumpy,gdalDType=gdal.GDT_Float32,npDType=np.float32):
        '''
        It writes the array (numpy.ndarray) in memory into the disk.
        :param outputPath: the file in the disk to be save.
        :param nbands: the number of channels (bands)
        :param proj: It can be None.
        :param trans: It can be None.
        :param dataNumpy: the array to be saved.
        :param gdalDType: The data type to store in the disk represented by gdalDType. \
               There is s map between the gdalDType and the data type of numpy. It starts with gdal.GDT_* \.
        :param npDType: the data type of numpy in the memory.
        :return: None
        '''
        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            return
        if len(dataNumpy.shape) == 3:
            (height, width) = (dataNumpy.shape[1], dataNumpy.shape[2])
        else:
            (height, width) = dataNumpy.shape
        out_data_set = driver.Create(outputPath, width, height, nbands, gdalDType)
        if not proj is None:
            out_data_set.SetProjection(proj)
        if not trans is None:
            out_data_set.SetGeoTransform(trans)
        for idx in range(nbands):
            out_band = out_data_set.GetRasterBand(idx + 1)
            if nbands > 1:
                data = np.asarray(dataNumpy[idx, ...], dtype=npDType)
            else:
                data = np.asarray(dataNumpy[...], dtype=npDType)
            out_band.WriteArray(data)