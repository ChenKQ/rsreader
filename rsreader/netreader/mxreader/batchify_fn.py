import numpy as np
from mxnet import ndarray as nd

def segnet_batchify_fn(data_lst):
    '''
    Use MIMOSegReader as Dataset which provides images with the shape of (C,H,W)
    and label with the shape of (H,W)
    Use DataLoader as the loader.
    :param data_lst: a list including data and label
    :return:
    '''
    img_lst = []
    lb_lst = []
    for idx, data in enumerate(data_lst):
        img, label = data # c, h, w
        top = img['data']
        img_lst.append(top)
        if not label['softmax_label'] is None:
            lb = label['softmax_label']
            lb_lst.append(lb)
    img_array = np.asarray(img_lst)
    img_array = nd.array(img_array,dtype=img_array.dtype)
    if not len(lb_lst)==0:
        lb_array = np.asarray(lb_lst)
        lb_array = nd.array(lb_array,dtype=lb_array.dtype)
    else:
        lb_array = None
    return [img_array, lb_array]