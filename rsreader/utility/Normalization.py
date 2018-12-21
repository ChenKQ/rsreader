from sklearn import preprocessing

class StdNorm(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return preprocessing.scale(args[0],**kwargs)

class MinMaxNorm(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return preprocessing.minmax_scale(args[0],**kwargs)

class SubtractMean(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, *args, **kwargs):
        img = args[0]
        for idx,channel in enumerate(img):
            img[idx] -= self.mean[idx]
        return img

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, *args, **kwargs):
        img = args[0]
        for idx,channel in enumerate(img):
            if not isinstance(self.mean, list):
                mean = self.mean
                std = self.std
            else:
                mean = self.mean[idx]
                std = self.std[idx]
            img[idx] -= mean
            img[idx] /= std
        return img