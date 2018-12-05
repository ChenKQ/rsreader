import numpy as np
from .SegReader import SegReader

class SCNNReader(SegReader):
    '''
    It corresponds to the network mentioned in the work of \
    $Semantic Segmentation of Aerial Images with Shuffling Convolutional Neural Networks$, K. Chen.
    '''
    def __init__(self,flist_name, data_root,
                 batchsize,cropsize, step, samplerate,cell,
                 img_trans,gt_trans,joint_trans,
                 withgt=True,
                 bandlist=None,
                 sampleseed = -1,
                 lvreadertype='pil',
                 parsertype = 'common',
                 openfirstly = False):
        self.samplerate = samplerate
        self.cell = cell
        super(SCNNReader, self).__init__(flist_name, data_root,
                                            batchsize, cropsize, step,
                                            img_trans, gt_trans, joint_trans,
                                            withgt=withgt,
                                            bandlist=bandlist,
                                            sampleseed=sampleseed,
                                            lvreadertype=lvreadertype,
                                            parsertype=parsertype,
                                            openfirstly=openfirstly)

    def read_img(self,sample,initx=None,inity=None):
        img,label = super(SCNNReader,self).read_img(sample,initx,inity)
        if not label is None:
            label = label[::self.cell, ::self.cell]
            reshaped_label = np.zeros((label.shape[0] * label.shape[1],))
            r = self.samplerate/self.cell
            step = label.shape[0]*label.shape[1]/(r*r)
            count = 0
            for h in range(r):
                for w in range(r):
                    subarea = label[h::r,w::r]
                    reshaped_label[count*step:(count+1)*step] = subarea.reshape((subarea.shape[0]*subarea.shape[1]))
                    count += 1
        else:
            reshaped_label = None
        return img,reshaped_label