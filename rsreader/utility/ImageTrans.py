from .LabelViz import LabelViz

class ImageTrans(object):

    def __init__(self,trans=[]):
        self.trans = trans

    def __call__(self, *args, **kwargs):
        img = args[0]
        for t in self.trans:
            img = t(img,**kwargs)
        return img

class VizLabel(LabelViz):
    def __init__(self, tempfile):
        super(VizLabel).__init__(tempfile)

    def __call__(self, *args, **kwargs):
        return super(VizLabel).unViz(args[0],hwcorder=False)