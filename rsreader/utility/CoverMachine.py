class CoverMachine(object):
    '''
    It is a class that computes the positions of the patches to read from the original image \
    and write to the prediction result according to a fixed patch size and step.
    '''
    def __init__(self, image_shape, patchsize, dst_size=-1, step=-1):
        '''
        Constructor.
        :param image_shape: The shape of the image with the shape of (height, width).
        :param patchsize: The size of each crop (patch) in the shape of (height, width).
        :param dst_size: The central part with the size of dst_size are cropped from the patch cropped from the image.
        :param step: The step to go. If step equals to -1 or patchsize, there is no overlap.
        '''
        self.image_shape = image_shape
        if not type(patchsize) is tuple:
            self.patchsize = (patchsize, patchsize)
        else:
            self.patchsize=patchsize
        if self.patchsize[0] == -1:
            self.patchsize = self.image_shape

        if not type(dst_size) is tuple:
            self.dst_size = (dst_size,dst_size)
        else:
            self.dst_size=dst_size
        if self.dst_size[0] == -1:
            self.dst_size = self.patchsize

        if not type(step) is tuple:
            self.step=(step,step)
        else:
            self.step = step
        if self.step[0] == -1:
            self.step = self.patchsize
        self.__preconfig()
        self.count = 0
        self.writeAreas = None
        self.readAreas = None
        self.calAreas()

    def __preconfig(self):
        self.pad=((self.patchsize[1]-self.dst_size[1])//2,(self.patchsize[0]-self.dst_size[0])//2)
        res = self.image_shape[1]-self.patchsize[1]+2*self.pad[1]
        if res%self.step[1] ==0:
            self.xl = res//self.step[1]
        else:
            self.xl=res//self.step[1]+1
        res = self.image_shape[0]-self.patchsize[0]+2*self.pad[0]
        if res%self.step[0] == 0:
            self.yl = res//self.step[0]
        else:
            self.yl=res//self.step[0]+1
        self.overlap=(max(self.dst_size[1]-self.step[1],0),max(self.dst_size[0]-self.step[0],0))
        self.rowidx=0
        self.colidx=-1
        self.cursor=-1

    def __len__(self):
        return len(self.readAreas)

    def calAreas(self):
        '''
        This method computes all the areas to read from the original image and to write to the prediction result.
        :return: None
        '''
        self.writeAreas = []
        self.readAreas = []
        while self.__shouldGo():
            writearea = self.write_area()
            self.writeAreas.append(writearea)
            readarea = self.calReadAreaLUP()
            self.readAreas.append(readarea)  # [(starty,startx),(starty,startx)...]
            self.count += 1
        self.reset()

    def __shouldGo(self):
        if not (self.colidx == self.xl and self.rowidx == self.yl):
            # self.cursor +=1
            if self.colidx == self.xl:
                self.colidx = 0
                self.rowidx += 1
            else:
                self.colidx += 1
            return True
        else:
            return False

    def next_idx(self):
        if self.cursor>=self.count-1:
            return False
        return True

    def calReadAreaLUP(self):
        '''
        It computes the areas to read from the original image in the current step.
        :return: the start point (left-upper point) of the area to read in the current step.
        '''
        if self.colidx!=self.xl:
            startx=self.step[1]*self.colidx-self.pad[1]
        else:
            startx=self.image_shape[1]+self.pad[1]-self.patchsize[1]
        if self.rowidx!=self.yl:
            starty=self.step[0]*self.rowidx-self.pad[0]
        else:
            starty=self.image_shape[0]+self.pad[0]-self.patchsize[0]
        return (starty,startx)

    def reset(self):
        self.rowidx=0
        self.colidx=-1
        self.cursor=-1

    def write_area(self):
        '''
        It computes the area to write to the prediction result in the current step.
        :return: [the areas in the prediction result, the areas cropped from the patch] in the current step.
        '''
        if self.colidx==0:
            startx=0
            endx=self.dst_size[1]-self.overlap[1]//2
            fromstartx=0
            fromendx=self.dst_size[1]-self.overlap[1]//2
        elif self.colidx==self.xl:
            localoverlap=(self.step[1]*(self.colidx-1)+self.dst_size[1])-(self.image_shape[1]-self.dst_size[1])
            startx=self.image_shape[1]-self.dst_size[1]+localoverlap//2
            endx=self.image_shape[1]
            fromstartx=localoverlap//2
            fromendx=self.dst_size[1]
        else:
            startx=self.step[1]*self.colidx+self.overlap[1]//2
            endx=self.step[1]*self.colidx+self.dst_size[1]-self.overlap[1]//2
            fromstartx=self.overlap[0]//2
            fromendx=self.dst_size[1]-self.overlap[1]//2

        if self.rowidx==0:
            starty=0
            endy=self.dst_size[0]-self.overlap[0]//2
            fromstarty=0
            fromendy=self.dst_size[0]-self.overlap[0]//2
        elif self.rowidx==self.yl:
            localoverlap=(self.step[0]*(self.rowidx-1)+self.dst_size[0])-(self.image_shape[0]-self.dst_size[0])
            starty=self.image_shape[0]-self.dst_size[0]+localoverlap//2
            endy=self.image_shape[0]
            fromstarty=localoverlap//2
            fromendy=self.dst_size[0]
        else:
            starty=self.step[0]*self.rowidx+self.overlap[0]//2
            endy=self.step[0]*self.rowidx+self.dst_size[0]-self.overlap[0]//2
            fromstarty=self.overlap[0]//2
            fromendy=self.dst_size[0]-self.overlap[0]//2
        return [(starty,endy,startx,endx),(fromstarty,fromendy,fromstartx,fromendx)]


if __name__ == '__main__':
    cover = CoverMachine(image_shape=(6000,6000),patchsize=(224,224),dst_size=(224,224),step=(200,200))
    print('length: ', len(cover))
    print(len(cover.writeAreas), len(cover.readAreas))
    for i in range(0,len(cover)):
        print(cover.readAreas[i], cover.writeAreas[i])