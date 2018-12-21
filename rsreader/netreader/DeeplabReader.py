from .SegReader import SegReader
import cv2

class DeeplabReader(SegReader):
    def __init__(self,flist_name, data_root,
                 batchsize,cropsize,step,samplerate,
                 img_trans,gt_trans,joint_trans,
                 targetSize=None,
                 withgt=True,
                 bandlist=None,
                 sampleseed = -1,
                 lvreadertype='pil',
                 parsertype = 'common',
                 openfirstly = False):
        self.samplerate = samplerate
        self.targetSize = targetSize
        super(DeeplabReader,self).__init__(flist_name, data_root,
                 batchsize,cropsize,step,
                 img_trans,gt_trans,joint_trans,
                 withgt=withgt,
                 bandlist=bandlist,
                 sampleseed = sampleseed,
                 lvreadertype=lvreadertype,
                 parsertype = parsertype,
                 openfirstly = openfirstly)

    def read_img(self,sample,initx=None,inity=None):
        img, label = super(DeeplabReader, self).read_img(sample, initx, inity)
        if self.targetSize is None and not self.samplerate is None:
            label = label[::self.samplerate, ::self.samplerate]
        if not self.targetSize is None:
            label = cv2.resize(label, dsize=self.targetSize, interpolation=cv2.INTER_NEAREST)
        return img,label

