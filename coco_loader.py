from scipy.io import loadmat
from loader import Loader, DATASETS_ROOT

from pycocotools.coco import COCO
from pycocotools import mask

import cv2
import numpy as np

COCO_VOC_CATS = ['__background__', 'airplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
                 'dog', 'horse', 'motorcycle', 'person', 'potted plant',
                 'sheep', 'couch', 'train', 'tv']

COCO_NONVOC_CATS = ['apple', 'backpack', 'banana', 'baseball bat',
                    'baseball glove', 'bear', 'bed', 'bench', 'book', 'bowl',
                    'broccoli', 'cake', 'carrot', 'cell phone', 'clock', 'cup',
                    'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee',
                    'giraffe', 'hair drier', 'handbag', 'hot dog', 'keyboard',
                    'kite', 'knife', 'laptop', 'microwave', 'mouse', 'orange',
                    'oven', 'parking meter', 'pizza', 'refrigerator', 'remote',
                    'sandwich', 'scissors', 'sink', 'skateboard', 'skis',
                    'snowboard', 'spoon', 'sports ball', 'stop sign',
                    'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
                    'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
                    'truck', 'umbrella', 'vase', 'wine glass', 'zebra']

COCO_CATS = COCO_VOC_CATS+COCO_NONVOC_CATS

coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
            67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
            38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
            81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
            42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
            80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
            7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
            46, 'zebra': 24}
coco_ids_to_cats = dict(map(reversed, list(coco_ids.items())))


class COCOLoader(Loader):
    cats_to_ids = dict(map(reversed, enumerate(COCO_CATS)))
    ids_to_cats = dict(enumerate(COCO_CATS))
    num_classes = len(COCO_CATS)
    categories = COCO_CATS[1:]

    def __init__(self, year, proposals, split, num_proposals=2000, excluded=[], cats=COCO_CATS):
        super().__init__()
        # TODO support cat reshuffling
        self.dataset = 'coco'
        self.coco_ids_to_internal = {k: self.cats_to_ids[v] for k, v in coco_ids_to_cats.items()}
        self.ids_to_coco_ids = dict(map(reversed, self.coco_ids_to_internal.items()))
        self.split = split + year
        assert self.split in ['train2014', 'val2014', 'test2014', 'test2015']
        self.root = DATASETS_ROOT + 'coco/'
        assert proposals in ['mcg', 'edgeboxes']
        self.proposals = proposals
        self.num_proposals = num_proposals
        assert num_proposals >= 0
        if excluded == []:
            self.included_coco_ids = list(coco_ids.values())
        else:
            included_internal_ids = [i for i in self.coco_ids_to_internal.values()
                                     if i not in excluded]
            self.included_coco_ids = [coco_ids[COCOLoader.ids_to_cats[i]]
                                      for i in included_internal_ids]

        self.coco = COCO('%s/annotations/instances_%s.json' % (self.root, self.split))

    def load_image(self, img_id, resize=True):
        img = self.coco.loadImgs(img_id)[0]
        im = cv2.imread('%simages/%s/%s' % (self.root, self.split, img['file_name']))
        return self.convert_and_maybe_resize(im, resize)

    def read_proposals(self, img_id):
        img = self.coco.loadImgs(img_id)[0]
        name = img['file_name'][:-4]
        if self.proposals == 'edgeboxes':
            mat = loadmat('%sEdgeBoxesProposals/%s/%s.mat' % (self.root, self.split, name))
            # mat = loadmat('%sEdgeBoxesProposalsSmall/%s/%s.mat' % (self.root, self.split, name))
            bboxes = mat['bbs'][:, :4]
        if self.proposals == 'selective_search':
            raise NotImplementedError
        if self.proposals == 'mcg':
            mat = loadmat('%sMCGProposals/MCG-COCO-%s-boxes/%s.mat' % (self.root, self.split, name))
            bboxes = mat['boxes']
            # (y1, x1, y2, x2) -> (x1, y1, w, h)
            y1 = bboxes[:, 0]
            x1 = bboxes[:, 1]
            y2 = bboxes[:, 2]
            x2 = bboxes[:, 3]
            bboxes = np.stack([x1, y1, x2-x1, y2-y1], axis=1)
            # print(bboxes.shape)
        if self.num_proposals == 0:
            return bboxes
        else:
            return bboxes[:self.num_proposals]

    def get_filenames(self):
        # strictly speaking those are not filenames,
        # but the usage is consistent in this class
        return self.coco.getImgIds()

    def get_coco_annotations(self, img_id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(
            imgIds=img_id, catIds=self.included_coco_ids, iscrowd=False))
        return anns

    # TODO should I filter categories here?
    def read_annotations(self, img_id):
        anns = self.get_coco_annotations(img_id)

        bboxes = [ann['bbox'] for ann in anns]
        cats = [coco_ids_to_cats[ann['category_id']] for ann in anns]
        labels = [COCOLoader.cats_to_ids[cat_name] for cat_name in cats]

        img = self.coco.loadImgs(img_id)[0]

        return np.round(bboxes).astype(np.int32).reshape((-1, 4)), np.array(labels),\
            img['width'], img['height'], np.zeros_like(labels, dtype=np.bool)

    def _read_segmentation(self, ann, H, W):
        s = ann['segmentation']
        s = s if type(s) == list else [s]
        return mask.decode(mask.frPyObjects(s, H, W)).max(axis=2)


def print_classes_stats(ids, title=""):
        imgs = set()
        for s in ids[:20]:
            imgs = imgs | s
        print("{}: {}".format(title, len(imgs)))


if __name__ == '__main__':
        root = DATASETS_ROOT + 'coco/'
        split = 'train2014'
        coco = COCO('%s/annotations/instances_%s.json' % (root, split))
        ids = []
        for cat in COCO_CATS[1:]:
            imgs = coco.getImgIds(catIds=[coco_ids[cat]])
            print("{}: {}".format(cat, len(imgs)))
            ids.append(set(imgs))
        print_classes_stats(ids[:20], "VOC")
        print_classes_stats(ids[20:], "NONVOC")
        print_classes_stats(ids[20:30], "10 cats after VOC")
