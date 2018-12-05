from mxnet.io import DataIter
import mxnet as mx
from ..SimpleBatch import SimpleBatch
import random

class NetReader(DataIter):
    '''
    Deprecated as the gluon is becoming increasing popular. It is a reader for networks in mxnet. \
    We do not maintain this class anymore.
    '''
    def __init__(self, netreader, niter, mode='random'):
        '''
        Constructor.
        :param netreader: Netreader.
        :param niter: The number of iterations per epoch.
        :param mode: 'random' or 'shuffle' or 'order'.
        '''
        self.netreader = netreader
        self.niter = niter
        self.cursor = 0
        super(DataIter,self).__init__()
        self.sampleseed = self.netreader.sampleseed
        self.batchsize = self.get_batch_size()
        self.data_name = self.netreader.data_name
        self.label_name = self.netreader.label_name
        self.withgt = self.netreader.withgt
        self.data, self.label = self.read(self.sampleseed, None, None)
        self.mode = mode
        if not self.mode == 'random':
            self.ncrop = len(self.netreader)
            self.niter = self.ncrop / self.batchsize
            self.readorder = range(0,self.ncrop)
        if self.mode == 'shuffle':
            self.readorder = random.shuffle(self.readorder)

    def readOnIdxes(self, idx_list):
        return self.netreader.readOnIdxes(idx_list)

    def read(self, sampleidx, initx, inity):
        return self.netreader.read(sampleidx, initx, inity)

    def __getitem__(self, idx):
        return self.netreader[idx]

    def read_img(self,sample,initx=None,inity=None):
        return self.netreader.read_img(sample,initx,inity)

    def __len__(self):
        return self.niter

    @property
    def provide_data(self):
        return [(k,tuple([self.batchsize] + list(v.shape[1:]))) for k,v in self.data.items()]

    @property
    def provide_label(self):
        if self.label.items()[0][1] is None:
            return None
        return [(k,tuple([self.batchsize] + list(v.shape[1:]))) for k,v in self.label.items()]

    def get_batch_size(self):
        return self.netreader.batchsize

    def reset(self):
        self.cursor = 0
        if self.mode == 'shuffle':
            self.readorder = random.shuffle(self.readorder)

    def iter_next(self):
        self.cursor += 1
        if self.cursor <= self.niter:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            if self.mode == 'random':
                self.data,self.label = self.read(self.sampleseed,None,None)
            else:
                self.data, self.label = self.readOnIdxes(self.readorder[(self.cursor-1)*self.batchsize:self.cursor*self.batchsize])
            if self.withgt:
                return SimpleBatch([mx.nd.array(self.data[n]) for n in self.data_name],
                                   [mx.nd.array(self.label[n]) for n in self.label_name])
            else:
                return SimpleBatch([mx.nd.array(self.data[n]) for n in self.data_name], None)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()