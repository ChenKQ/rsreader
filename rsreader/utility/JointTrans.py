from .DataAugmentation import DataAugument

class JointTrans(DataAugument):
    def __init__(self, funcs=None, docopy = False):
        super(JointTrans,self).__init__(funcs=funcs)
        self.docopy = docopy

    def __call__(self, *args, **kwargs):
        img = args[0]
        label = args[1]
        img, label = self.aug(img, label)
        if self.docopy is True:
            img = img.copy()
            label = label.copy()
        return img,label