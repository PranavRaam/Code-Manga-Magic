import numpy as np
import cv2
from skimage.io import imread


def variable_to_cv2_image(varim):
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def normalize(data):
    return np.float32(data/255.)

def remove_dataparallel_wrapper(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict

def is_rgb(im_path):
    rgb = False
    im = imread(im_path)
    if (len(im.shape) == 3):
        if not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])):
            rgb = True
    print("rgb: {}".format(rgb))
    print("im shape: {}".format(im.shape))
    return rgb
