import numpy as np
from .SegReader import SegReader

class MIMOSegReader(SegReader):
    def __init__(self,flist_name, data_root,
                 batchsize,cropsize,step,samplerate,cell,
                 img_trans,gt_trans,joint_trans,
                 withgt=True,
                 bandlist=None,
                 sampleseed = -1,
                 data_name=['data'],
                 label_name=['softmax_label'],
                 lvreadertype='pil',
                 parsertype = 'common',
                 openfirstly = False):
        self.samplerate = samplerate
        self.cell = cell
        self.data_name = data_name
        self.label_name = label_name
        super(SegReader, self).__init__(flist_name, data_root,
                                            batchsize, cropsize, step,
                                            img_trans, gt_trans, joint_trans,
                                            withgt=withgt,
                                            bandlist=bandlist,
                                            sampleseed=sampleseed,
                                            lvreadertype=lvreadertype,
                                            parsertype=parsertype,
                                            openfirstly=openfirstly)

    def readOnIdxes(self,idx_list):
        '''
        It read the image based on the index.
        :param idx_list: The list of the indexes.
        :return: (data, label)
        '''
        data={}
        label={}
        for item in self.data_name:
            data[item]=[]
        for item in self.label_name:
            label[item]=[]
        for idx in idx_list:
            data_,label_=self.__getitem__(idx)
            for k,v in data_.items():
                data[k].append(v)
            for k,v in label_.items():
                if not v is None:
                    label[k].append(v)
        for item in self.data_name:
            data[item] = np.asarray(data[item])
        for item in self.label_name:
            if len(label[item]) == 0:
                label[item]=None
            else:
                label[item]=np.asarray(label[item])
        return data,label

    def read(self,sampleidx,initx,inity):
        '''
        This methods reads several patches from several samples.
        :param sampleidx: The index of the sample to be read. If it is -1, then the sample will be chosen by random.
        :param initx: The x-coordinate of the left-upper point of the crop in the image.
        :param inity: The y-coordinate of the left-upper point of the crop in the image.
        :return: (img, label)
        '''
        data={}
        label={}
        for item in self.data_name:
            data[item]=[]
        for item in self.label_name:
            label[item]=[]
        for i in range(self.batchsize):
            sample = self.readerStore.getOneSample(sampleidx)
            data_,label_=self.read_img(sample,initx,inity)
            for k,v in data_.items():
                data[k].append(v)
            for k,v in label_.items():
                if not v is None:
                    label[k].append(v)
        for item in self.data_name:
            data[item] = np.asarray(data[item])
        for item in self.label_name:
            if len(label[item]) == 0:
                label[item]=None
            else:
                label[item]=np.asarray(label[item])
        return data,label

    def read_img(self,sample,initx=None,inity=None):
        '''
        This methods reads several patches from several samples.
        :param sampleidx: The index of the sample to be read. If it is -1, then the sample will be chosen by random.
        :param initx: The x-coordinate of the left-upper point of the crop in the image.
        :param inity: The y-coordinate of the left-upper point of the crop in the image.
        :return: (img, label)
        '''
        img,label = super(MIMOSegReader,self).read_img(sample,initx,inity)
        return dict([(self.data_name[0], img)]), dict([(self.label_name[0], label)])