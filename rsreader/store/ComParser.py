from ..lvreader.BaseReader import BaseReader
from collections import namedtuple
import os

Sample = namedtuple('Sample',['top','gt','nchannel'])

class ComParser(object):
    '''
    This is a class that parses each line into sevel readers.
    '''
    def __init__(self,data_root,withgt,lvreadertype='gdal'):
        '''
        Constructor.
        :param data_root: The root directory that stores the data.
        :param withgt: If each line contains ground-truth (gt).
        :param lvreadertype: which low-level reader is used to read the data. In default, it is GdalReader
        '''
        self.data_root = data_root
        self.withgt = withgt
        self.lvreadertype = lvreadertype

    def parseLine2Sample(self,line):
        '''
        This method parses each line into readers, which are represented as one sample. \
        Each sample is stored in a dictionary which contains three keys including 'top', 'gt' and 'nchannel'.
        :param line: each line represents the images and the corresponding ground truth.
        :return: A sample represented as  a dictionary.
        '''
        files = line.strip('\n').strip(' ').split(' ')
        lenfiles=len(files)
        nchannel=0
        topreaders = []
        gtreaders = []
        for idx,f in enumerate(files):
            if self.withgt and idx==( lenfiles-1):
                gtreaders.append(BaseReader(os.path.join(self.data_root,f),readertype=self.lvreadertype))
            else:
                topreaders.append(BaseReader(os.path.join(self.data_root, f),readertype=self.lvreadertype))
                nchannel += topreaders[idx].getNChannel()
        sample = Sample(topreaders, gtreaders, nchannel)
        return sample