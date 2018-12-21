import numpy as np
import cv2
import os
from ..lvreader.BaseReader import BaseReader

class LabelViz(object):
    '''
    This class is used to convert formats between color maps which is used for visualization and label map which has only label like 0,1,2,3,....
    '''
    def __init__(self,tempfile):
        '''
        Initialization Method.
        :param tempfile: The template file.
        '''
        self.tempfile = tempfile
        with open(self.tempfile,'r') as f:
            lines=f.readlines()
        self.mapping=[]
        for line in lines:
            self.mapping.append(self.str2List(line))

    def str2List(self,s):
        '''
        This method converts from '255,255,0:2'->'255,255,0','2'->['255','255','0'],'2'->[255,255,0],1->{'color':[255,255,0], 'label':1}
        :param s: '255,255,0:2'
        :return: {'color':[255,255,0], 'label':1} . Color will be a list and label will be a number.
        '''
        color,lb=s.strip('\n').split(':')
        lb=int(lb)
        color=color.split(',')
        channels=[]
        for channel in color:
            channels.append(int(channel))
        mapping={}
        mapping['color']=channels
        mapping['label']=lb
        return mapping
    def unViz(self,colorMap,hwcorder=True):
        '''
        This method is used to convert from color map to label map
        :param colorMap: color map matrix(one channel used for two classes and three channels for multi-class are all right). Its shape should be (c,h,w)
        :return: label map (only one channel)
        '''
        if len(colorMap.shape) ==3:
            assert len(self.mapping[0]['color'])==3
            if hwcorder:
                colormaphwc = colorMap
            else:
                colormaphwc=np.swapaxes(colorMap,0,2)  ##c,h,w -->w,h,c
                colormaphwc=np.swapaxes(colormaphwc,0,1)  ##w,h,c -->h,w,c
            labelmap=np.zeros(colormaphwc.shape[0:2],dtype=np.uint8)
            for item in self.mapping:
                color=item['color']
                lb=item['label']
                idxtags = (colormaphwc==color)
                labelmap += (idxtags[...,0]*idxtags[...,1]*idxtags[...,2])*np.ones(idxtags.shape[0:2],dtype=np.uint8)*lb
            return labelmap
        else:
            assert len(self.mapping[0]['color'])==1
            labelmap = np.zeros(colorMap.shape, dtype=np.uint8)
            for item in self.mapping:
                color=item['color']
                lb=item['label']
                labelmap += (colorMap==color[0])*np.ones(colorMap.shape,dtype=np.uint8)*lb
            return labelmap
    def Viz(self,labelMap):
        '''
        This method is used to convert from label map to color map, it is used for visualization
        :param labelMap: label map matrix(one channel should be provided)
        :return: color map matrix which can be saved as three channel picture is returned. Its shape should be (h,w,c)
        '''
        colormap= np.zeros((labelMap.shape[0],labelMap.shape[1],3),dtype=np.uint8)
        for item in self.mapping:
            color=item['color']
            label=item['label']
            if len(color)==1:
                colormap[...,0] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
                colormap[...,1] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
                colormap[...,2] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
            else:
                assert len(color)==3
                colormap[...,0] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[0]
                colormap[...,1] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[1]
                colormap[...,2] += (labelMap==label)*np.ones(colormap.shape[0:2],dtype=np.uint8)*color[2]
        return colormap

    def convert2Lable(self,colormapfile,saveFolder):
        '''
        This method is used to convert one color map file to label file, which is saved to saveFolder
        :param colormapfile: a file of color map
        :param saveFolder: destination save folder
        :return:
        '''
        reader = BaseReader(colormapfile)
        colormap=reader.readImg()
        labelmap = self.unViz(colormap,hwcorder=False)
        basename=os.path.basename(colormapfile)
        cv2.imwrite(os.path.join(saveFolder,basename),labelmap)

    def convert2Color(self,labelfile,saveFolder):
        '''
        This method is used to convert one label map file to color map file, which is save to saveFolder.
        :param labelfile: a file of label map
        :param saveFolder:  destination save folder
        :return:
        '''
        reader=BaseReader(labelfile)
        labelmap=reader.readImg()[0]
        colormap=self.Viz(labelmap)
        basename=os.path.basename(labelfile)
        colorsave=np.zeros(colormap.shape,dtype=np.uint8)
        colorsave[...,0],colorsave[...,1],colorsave[...,2]=colormap[...,2],colormap[...,1],colormap[...,0]
        cv2.imwrite(os.path.join(saveFolder,basename),colorsave)

    def compare(self,map1,map2):
        img1=cv2.imread(map1,-1)
        img2=cv2.imread(map2,-1)
        if len(img1.shape) != len(img2.shape):
            print(np.sum(img1-img2[...,0]))
        else:
            print(np.sum(img1-img2))