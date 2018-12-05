import numpy as np
from ..store.SampleStore import SampleStore
from ..utility.CoverMachine import CoverMachine


class SegReader(object):
    '''
    It is a class that read images and labels.
    '''
    def __init__(self, flist_name, data_root,
                 batchsize,cropsize,step,
                 img_trans,gt_trans,joint_trans,
                 withgt=True,
                 bandlist=None,
                 sampleseed = -1,
                 lvreadertype='pil',
                 parsertype = 'common',
                 openfirstly = False):
        '''
        Constructor.
        :param flist_name: The file name of the list of samples.
        :param data_root: The root directory of the data.
        :param batchsize: The number of samples in a batch.
        :param cropsize: The size of each crop.
        :param step: The step when cropping from the original image.
        :param img_trans: Transformation to the original image.
        :param gt_trans: Transformation to the ground-truth.
        :param joint_trans: Joint transformation to the image and the ground-truth.
        :param withgt: If ground truth is included in the list of samples.
        :param bandlist: The selected bands to be used.
        :param sampleseed: The seed of the sample.
        :param lvreadertype: Which low-level reader is used.
        :param parsertype: Which parser is used to parse each line of the list file.
        :param openfirstly: If it is set as true, it will open all the images firstly. Or it will read the image temporarily.
        '''
        self.flist_name = flist_name
        self.data_root  = data_root
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.step = step  #it is only used for computing the read areas
        self.img_trans = img_trans
        self.gt_trans = gt_trans
        self.joint_trans = joint_trans
        self.withgt = withgt
        self.bandlist = bandlist
        self.sampleseed = sampleseed
        self.lvreadertype = lvreadertype
        self.parsertype = parsertype
        self.openfirstly = openfirstly
        self.readerStore=SampleStore(self.flist_name,self.data_root,self.withgt,
                                     openfirstly=self.openfirstly,lvreadertype=self.lvreadertype,parsertype=self.parsertype)
        if not type(self.cropsize) is tuple:
            self.cropsize = (self.cropsize,self.cropsize)
        self.read_areas=[]
        self.read_lengths=[]
        self.__getLoadIndex()

    def __getLoadIndex(self):
        for idx in range(0,len(self.readerStore.lines)):
            sample = self.readerStore.getOneSample(idx)
            image_shape = sample.top[0].getSize()
            cover = CoverMachine(image_shape=image_shape,patchsize=self.cropsize,
                                     dst_size=self.cropsize, step = self.step)
            readorder = cover.readAreas
            self.read_areas.append(readorder)
            self.read_lengths.append(len(readorder))

    def __len__(self):
        return sum(self.read_lengths)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        assert self.__getScale() == [1.0, 1.0]
        imgidx = 0
        l = self.read_lengths[0]
        while True:
            if idx>=l:
                imgidx +=1
                idx -= l
                l = self.read_lengths[imgidx]
            else:
                patchidx = self.read_areas[imgidx][idx]
                break
        sample = self.readerStore.getOneSample(imgidx)
        img,label = self.read_img(sample,patchidx[1],patchidx[0])
        return img,label


    def readOnIdxes(self,idx_list):
        '''
        It read the image based on the index.
        :param idx_list: The list of the indexes.
        :return: (data, label) where the data has the shape of (batch, channel, height, width)
        '''
        data=[]
        label=[]
        for idx in idx_list:
            data_,label_=self.__getitem__(idx)
            data = data.append(data_)
            if not label_ is None:
                label = label.append(label_)
        data = np.asarray(data)
        if not len(label)==0:
            label = np.asarray(label)
        else:
            label = None
        return data,label

    def __randomCrop(self,patchsize):
        return (int(round(np.random.rand()*(patchsize[0]-self.readsize[0]))),int(round(np.random.rand()*(patchsize[1]-self.readsize[1]))))

    def _randomCrop(self,patchsize):
        return self.__randomCrop(patchsize)

    def read(self,sampleidx,initx,inity):
        '''
        This methods reads several patches from several samples.
        :param sampleidx: The index of the sample to be read. If it is -1, then the sample will be chosen by random.
        :param initx: The x-coordinate of the left-upper point of the crop in the image.
        :param inity: The y-coordinate of the left-upper point of the crop in the image.
        :return: (img, label) where the img has the shape of (batch, channel, height, width)
        '''
        data=[]
        label=[]
        if not type(sampleidx) is list:
            sampleidx = [sampleidx] * self.batchsize
        for i in range(self.batchsize):
            sample = self.readerStore.getOneSample(sampleidx[i])
            data_,label_=self.read_img(sample,initx,inity)
            data.append(data_)
            if not label_ is None:
                label.append()
        data = np.asarray(data)
        if not len(label)==0:
            label = np.asarray(label)
        else:
            label = None
        return data,label

    def __getScale(self):
        scale = [1.0,1.0]
        if not self.joint_trans is None:
            gs = self.joint_trans.getScale()
        else:
            gs = (1,1)
        scale[0] = scale[0]*gs[0]
        scale[1] = scale[1] * gs[1]
        return scale

    def read_img(self,sample,initx=None,inity=None):
        '''
        This method reads a patch from the original image in the sample.
        :param sample: The sample to read from.
        :param initx: The x-coordinate of the left-upper point of the crop.
        :param inity: The y-coordinate of the left-upper point of the crop.
        :return: (img, label) where the img has the shape of (channel, height, width)
        '''
        imgReaders=sample.top
        if self.withgt:
            gtReader=sample.gt[0]
        else:
            gtReader = None
        if self.bandlist is None:
            nchannel= sample.nchannel
        else:
            nchannel = 0
            for cc in self.bandlist:
                nchannel += len(cc)
        scale = self.__getScale()
        self.readsize = (int(scale[0]*self.cropsize[0]),int(scale[1]*self.cropsize[1]))
        if self.readsize[0] == -1:
            self.readsize = imgReaders[0].getSize()
        img=np.zeros((nchannel,self.readsize[0],self.readsize[1]),dtype=np.float32)
        size = imgReaders[0].getSize()
        crop = self.__randomCrop(size)
        if initx is None:
            initx = crop[1]
        if inity is None:
            inity = crop[0]
        startchannel = 0
        for idx,reader in enumerate(imgReaders):
            if self.bandlist is None:
                nchannel = reader.getNChannel()
                bandlst = None
            else:
                nchannel = len(self.bandlist[idx])
                bandlst = self.bandlist[idx]
            img[startchannel:startchannel + nchannel, ...] = reader.readPatch(initx, inity,
                                                                                     width = self.readsize[1],
                                                                                     height=self.readsize[0],
                                                                                     bandlst=bandlst,
                                                                                     dtype=np.float32)
            startchannel += nchannel
        if self.withgt:
            label=gtReader.readPatch(initx,inity,width = self.readsize[1],height=self.readsize[0],
                                            bandlst=[],dtype=np.int64)[0]
            assert img.shape[1:] == label.shape
        else:
            label=None
        if not self.joint_trans is None:
            img,label = self.joint_trans(img,label)
        if not self.img_trans is None:
            img = self.img_trans(img)
        if not self.gt_trans is None:
            label = self.gt_trans(label)
        return img, label