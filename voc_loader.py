from scipy.io import loadmat
from loader import Loader, DATASETS_ROOT

import cv2
import numpy as np
import xml.etree.ElementTree as ET

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']


class VOCLoader(Loader):

    def __init__(self, year, proposals, split, num_proposals=2000, excluded=[],
                 cats=VOC_CATS):
        super().__init__()
        assert year in ['07', '12']
        self.dataset = 'voc'
        self.year = year
        self.root = DATASETS_ROOT + ('voc/VOCdevkit/VOC20%s/' % year)
        self.split = split
        assert split in ['train', 'val', 'trainval', 'test']
        self.proposals = proposals
        self.num_proposals = num_proposals
        assert num_proposals >= 0
        self.excluded_cats = excluded

        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]

    def load_image(self, name, resize=True):
        im = cv2.imread('%sJPEGImages/%s.jpg' % (self.root, name))
        out = self.convert_and_maybe_resize(im, resize)
        return out

    def get_filenames(self):
        with open(self.root+'ImageSets/Main/%s.txt' % self.split, 'r') as f:
            return f.read().split('\n')[:-1]

    def read_proposals(self, name):
        if self.proposals == 'edgeboxes':
            mat = loadmat('%sEdgeBoxesProposals/%s.mat' % (self.root, name))
            bboxes = mat['bbs'][:, :4]
        if self.proposals == 'selective_search':
            bboxes = np.load('%sSelectiveSearchProposals/%s.npy' % (self.root, name))
        if self.num_proposals == 0:
            return bboxes
        else:
            return bboxes[:self.num_proposals]

    def read_annotations(self, name, exclude=True):
        bboxes = []
        cats = []

        tree = ET.parse('%sAnnotations/%s.xml' % (self.root, name))
        root = tree.getroot()
        # image_path = images_dir+root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        difficulty = []
        for obj in root.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            difficult = (int(obj.find('difficult').text) != 0)
            difficulty.append(difficult)
            cats.append(cat)
            bbox_tag = obj.find('bndbox')
            x = int(bbox_tag.find('xmin').text)
            y = int(bbox_tag.find('ymin').text)
            w = int(bbox_tag.find('xmax').text)-x
            h = int(bbox_tag.find('ymax').text)-y
            bboxes.append((x, y, w, h))

        gt_cats = np.array(cats)
        gt_bboxes = np.array(bboxes)
        difficulty = np.array(difficulty)

        if exclude:
            incl_mask = np.array([cat not in self.excluded_cats for cat in gt_cats])
            gt_bboxes = gt_bboxes[np.logical_and(~difficulty, incl_mask)]
            gt_cats = gt_cats[np.logical_and(~difficulty, incl_mask)]
            difficulty = None
        return gt_bboxes, gt_cats, width, height, difficulty


def create_permutation(last_class):
    cats = list(VOC_CATS)
    i = cats.index(last_class)
    j = cats.index('tvmonitor')
    cats[i], cats[j] = cats[j], cats[i]
    cats[:-1] = sorted(cats[:-1])
    return cats


def class_stats(ids, start_id, end_id):
    common = set()
    for i in range(start_id, end_id+1):
        common = common | ids[i]
    print("Classes from {} to {} are in {} images".format(start_id, end_id, len(common)))


if __name__ == '__main__':
    print("Statistics per class: ")
    ids = {i: set() for i in range(1, 21)}
    loader = VOCLoader('07', 'edgeboxes', 'trainval')
    total = 0

    for name in loader.get_filenames():
        gt_cats = loader.read_annotations(name)[1]
        for cid in gt_cats:
            ids[cid].add(name)
    for i in ids.keys():
        print("%s: %i" % (VOC_CATS[i], len(ids[i])))
        total += len(ids[i])
    print("TOTAL: %i" % total)

    class_stats(ids, 1, 10)
    class_stats(ids, 11, 20)
