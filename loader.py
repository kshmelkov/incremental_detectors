import cv2

DATASETS_ROOT = './datasets/'


class Loader(object):
    def __init__(self):
        pass

    def convert_and_maybe_resize(self, im, resize):
        scale = 1.0
        if resize:
            h, w, _ = im.shape
            scale = min(1000/max(h, w), 600/min(h, w))
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255.0
        return im, scale
