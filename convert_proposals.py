from scipy.io import loadmat
import numpy as np

DATASETS_ROOT = '/home/lear/kshmelko/scratch/datasets/'


def read_selective_search(year, dataset):
    root = DATASETS_ROOT + ('voc/VOCdevkit/VOC20%s/' % year)
    matlab_file = loadmat('%sSelectiveSearchProposals/voc_20%s_%s.mat' % (root, year, dataset))
    images = matlab_file['images'].ravel()
    raw_data = matlab_file['boxes'].ravel()

    n = images.shape[0]
    for i in range(n):
        name = images[i][0]
        boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        keep = unique_boxes(boxes)
        boxes = boxes[keep, :]
        keep = filter_small_boxes(boxes, 16)
        boxes = boxes[keep, :]
        proposals = xyxy_to_xywh(boxes)
        np.save("%sSelectiveSearchProposals/%s.npy" % (root, name), proposals)


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (w > 0).all()
    assert (h > 0).all()
    assert (x1+w <= width).all()
    assert (y1+h <= height).all()


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h >= min_size))[0]
    return keep


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def write_proposals(name):
    pass


read_selective_search('07', 'test')
read_selective_search('07', 'trainval')
read_selective_search('12', 'test')
read_selective_search('12', 'train')
read_selective_search('12', 'val')
