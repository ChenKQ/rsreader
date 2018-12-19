from .SegReader import SegReader
import numpy as np

class CollectReader(SegReader):
    '''
    It reads images from ReaderStore and puts the images in a list like [image1, image2, ...], where all the images \
    has the shape of (channel, height, width). The shape of different images should be the same. \
    Different from the SegReader which concatenates different images in the channel dimension, this class just put \
    these images in a list. The further process can be done in other places. This function provides further possibilities \
    to do a wider range of operations.
    Attention: Only utility.Normalization.Normalization support img_trans by now. Please set joint_trans=None and gt_trans=None.
    '''
    def read_img(self,sample,initx=None,inity=None):
        '''
        This method reads a patch from the original image in the sample.
        :param sample: The sample to read from.
        :param initx: The x-coordinate of the left-upper point of the crop.
        :param inity: The y-coordinate of the left-upper point of the crop.
        :return: (img_lst, label) where the img has the shape of [image1, image2, ...] and each image has the shape of \
        (channel, height ,width).
        '''
        imgReaders = sample.top
        if self.withgt:
            gtReader=sample.gt[0]
        else:
            gtReader = None
        scale = self._getScale()
        self.readsize = (int(scale[0]*self.cropsize[0]),int(scale[1]*self.cropsize[1]))
        if self.readsize[0] == -1:
            self.readsize = imgReaders[0].getSize()
        img_lst = []
        size = imgReaders[0].getSize()
        crop = self._randomCrop(size)
        if initx is None:
            initx = crop[1]
        if inity is None:
            inity = crop[0]
        for idx,reader in enumerate(imgReaders):
            if self.bandlist is None:
                bandlst = None
            else:
                bandlst = self.bandlist[idx]
            img_lst[idx] = reader.readPatch(initx, inity,
                                                     width = self.readsize[1],
                                                     height=self.readsize[0],
                                                     bandlst=bandlst,
                                                     dtype=np.float32)
        if self.withgt:
            label=gtReader.readPatch(initx,inity,width = self.readsize[1],height=self.readsize[0],
                                            bandlst=[],dtype=np.uint8)[0]
        else:
            label=None
        if not self.joint_trans is None:
            img_lst,label = self.joint_trans(img_lst,label)
        if not self.img_trans is None:
            img_lst = self.img_trans(img_lst)
        if not self.gt_trans is None:
            label = self.gt_trans(label)
        return img_lst, label
