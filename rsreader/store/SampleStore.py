import random

class SampleStore(object):
    '''
    It is a class that stores readers.
    '''
    def __init__(self,lstfile,data_root,withgt,openfirstly=False,lvreadertype='pil',parsertype='common'):
        '''
        Constructor.
        :param lstfile: A file that lists all the images (and the corresponding ground truth is withgt is true).
        :param data_root: the root directory of the data.
        :param withgt: If each line contains the corresponding ground truth.
        :param openfirstly: If read all the images into memory at once. If true, the lvreadertype should be 'gdal'.
        :param lvreadertype: The low-level reader used to read images. \
                             It can be PILReader, OpenCVReader, GdalReader or other readers.
        :param storetype: The parser used to parse each line into readers, for example, ComParser.
        '''
        self.lstfile=lstfile
        self.withgt=withgt
        self.data_root=data_root
        self.openfirstly = openfirstly
        self.lvreadertype = lvreadertype
        self.parsertype = parsertype
        self.samples=[]
        self.lines = None
        self.parser = None
        self.parseLstfile()

    def parseLstfile(self):
        '''
        It parses the  list file into samples and save the samples in self.samples. Each sample contains several readers.
        :return: None
        '''
        with open(self.lstfile,'r') as f:
            self.lines=f.readlines()
        if self.openfirstly:
            for line in self.lines:
                sample = self.parseLine2Sample(line)
                self.samples.append(sample)
        else:
            self.samples = None

    def parseLine2Sample(self,line):
        '''
        It parses each line into readers based on the parsertype given in the constructor.
        :param line: the content of one single line in the list file.
        :return: A sample in dictionary.
        '''
        if self.parsertype == 'common':
            from .ComParser import ComParser
            self.parser = ComParser(self.data_root,self.withgt,lvreadertype = self.lvreadertype)
        else:
            self.parser = self.parsertype(self.data_root, self.withgt, self.lvreadertype)
        return self.parser.parseLine2Sample(line)

    def getOneSample(self,idx=-1):
        '''
        From all the samples, return one sample according to idx. If idx is -1, it will return a sample by random.
        :return: a Sample instance.
        '''
        if self.openfirstly:
            if idx == -1:
                return random.sample(self.samples,1)[0]
            else:
                return self.samples[idx]
        else:
            if idx == -1:
                line = random.sample(self.lines,1)[0]
            else:
                line = self.lines[idx]
            return self.parseLine2Sample(line)

    def getSomeSamples(self,no):
        '''
        From all samples, return #no Samples.
        :param no: (int) the number of Samples to get.
        :return: some samples in a list where each sample is a Sample
        '''
        samples=[]
        for i in range(no):
            samples.append(self.getOneSample())
        return samples