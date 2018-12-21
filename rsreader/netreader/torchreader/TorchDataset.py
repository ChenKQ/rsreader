from torch.utils.data import Dataset

class TorchDataset(Dataset):
    '''
    It is a iterator that can be feed into deep networks. It is designed for remote sensing scene and parse images with gdal.
    '''
    def __init__(self, netreader, niter=-1, mode='shuffle'):
        '''
        Constructor.
        :param netreader: A netreader.
        :param niter: The number of iterations per epoch.
        :param mode: 'random' or 'shuffle'.
        '''
        super(Dataset, self).__init__()
        self.netreader = netreader
        self.niter = niter
        self.mode = mode
        self.cropsize = self.netreader.cropsize


    def __len__(self):
        if self.mode == 'shuffle':
            return len(self.netreader)
        else:
            return self.niter

    def __getitem__(self, idx):
        if self.mode == 'shuffle':
            return self.netreader[idx]
        else:
            sample = self.netreader.readerStore.getOneSample(-1)
            return self.netreader.read_img(sample, None, None)


