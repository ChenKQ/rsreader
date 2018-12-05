import numpy as np
from .MIMOSegReader import MIMOSegReader

class DSCNNReader(MIMOSegReader):
    def __init__(self,flist_name, data_root,
                 batchsize,cropsize,step,samplerate,cell,
                 img_trans,gt_trans,joint_trans,
                 withgt=True,
                 bandlist=None,
                 sampleseed = -1,
                 data_name=['data'],
                 label_name=['fusion_0_softmax_label', 'fusion_1_softmax_label', 'fusion_finest_softmax_label'],
                 readertype='pil',
                 storetype = 'common',
                 openfirstly = False):
        self.samplerate = samplerate
        self.cell = cell
        super(DSCNNReader, self).__init__(flist_name, data_root,
                                            batchsize, cropsize, step,
                                            img_trans, gt_trans, joint_trans,
                                            withgt=withgt,
                                            bandlist=bandlist,
                                            sampleseed=sampleseed,
                                            data_name=data_name,
                                            label_name=label_name,
                                            readertype=readertype,
                                            storetype=storetype,
                                            openfirstly=openfirstly)

    def __reshape(self,label,samplerate,cell):
        if not label is None:
            label = label[::cell, ::cell]
            reshaped_label = np.zeros((label.shape[0] * label.shape[1],))
            r = samplerate/cell
            step = label.shape[0]*label.shape[1]/(r*r)
            count = 0
            for h in range(r):
                for w in range(r):
                    subarea = label[h::r,w::r]
                    reshaped_label[count*step:(count+1)*step] = subarea.reshape((subarea.shape[0]*subarea.shape[1]))
                    count += 1
        else:
            reshaped_label = None
        return reshaped_label

    def read_img(self,sample,initx=None,inity=None):
        img,label = super(DSCNNReader,self).read_img(sample,initx,inity)
        label = label.items()[0][1]
        ret_label = {}
        idx = 0
        for n,r in zip(self.label_name,self.samplerate):
            reshaped_label = self.__reshape(label,r,self.cell[idx])
            idx += 1
            ret_label[n] = reshaped_label
        return img,ret_label

