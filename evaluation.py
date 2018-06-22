from config import args
from pycocotools.cocoeval import COCOeval
from utils import rescale_bboxes
from tabulate import tabulate

import numpy as np
import logging
import progressbar
import pickle
import os
import json
import matplotlib.pyplot as plt

import tensorflow as tf

log = logging.getLogger()

AVERAGE = "AVERAGE"


#TODO refactor it to VOCEval and COCOEval with a common ancestor
class Evaluation(object):
    def __init__(self, net, loader, ckpt, conf_thresh=0.5, nms_thresh=0.3):
        self.net = net
        self.loader = loader
        self.gt = {}
        self.dets = {}
        self.ckpt = ckpt
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.show_img = False

    def evaluate_network(self, eval_first_n):
        filenames = self.loader.get_filenames()[:eval_first_n]
        self.gt = {cid: {} for cid in range(1, self.loader.num_classes)}
        self.dets = {cid: [] for cid in range(1, self.loader.num_classes)}

        start = 0
        cache = '%sEvalCache/%s_%i.pickle' % (self.loader.root, args.run_name,
                                              self.ckpt)
        if os.path.exists(cache) and not self.show_img:
            log.info("Found a partial eval cache: %s", cache)
            with open(cache, 'rb') as f:
                self.gt, self.dets, start = pickle.load(f)

        bar = progressbar.ProgressBar()
        for i in bar(range(start, len(filenames))):
            self.process_image(filenames[i], i)
            if i % 10 == 0 and i > 0 and not self.show_img:
                with open(cache, 'wb') as f:
                    pickle.dump((self.gt, self.dets, i), f, pickle.HIGHEST_PROTOCOL)
        if not self.show_img:
            log.debug("Cached eval results %s after the end", cache)
            with open(cache, 'wb') as f:
                pickle.dump((self.gt, self.dets, len(filenames)), f, pickle.HIGHEST_PROTOCOL)
        aps, m = self.compute_ap()
        return aps

    def compute_ap(self):
        aps = {}
        table = []
        for cid in range(1, self.net.num_classes+1):
            cat_name = self.loader.ids_to_cats[cid]
            rec, prec = self.eval_category(cid)
            if rec is None or prec is None:
                table.append((cat_name, 0.0))
            else:
                ap = voc_ap(rec, prec, self.loader.year == '07')*100
                aps[self.loader.ids_to_cats[cid]] = ap
                table.append((cat_name, ap))

        resa = np.array(list(aps.values()))
        old_classes = [aps.get(k, 0) for k in self.loader.categories[:10]]
        new_classes = [aps.get(k, 0) for k in self.loader.categories[10:]]
        all_classes = [aps.get(k, 0) for k in self.loader.categories]
        table.append((AVERAGE+" 1-10", np.mean(old_classes)))
        table.append((AVERAGE+" 11-20", np.mean(new_classes)))
        table.append((AVERAGE+" ALL", np.mean(all_classes)))
        mean_ap = np.mean(list(aps.values()))
        # table.append((AVERAGE, mean_ap))
        x = tabulate(table, headers=["Category", "mAP"],
                     tablefmt='orgtbl', floatfmt=".1f")
        log.info("\n"+x)
        return aps, mean_ap

    def process_image(self, name, img_id):
        img, scale = self.loader.load_image(name)
        gt_bboxes, gt_cats, _, _, difficulty = self.loader.read_annotations(name, exclude=False)
        proposals = self.loader.read_proposals(name)
        proposals = rescale_bboxes(proposals, scale)
        gt_bboxes = rescale_bboxes(gt_bboxes, scale)

        for cid in np.unique(gt_cats):
            mask = (gt_cats == cid)
            bbox = gt_bboxes[mask]
            diff = difficulty[mask]
            det = np.zeros(len(diff), dtype=np.bool)
            self.gt[cid][img_id] = {'bbox': bbox, 'difficult': diff, 'det': det}

        det_cats, det_probs, det_bboxes = self.net.detect(img, proposals,
                                                          conf_thresh=self.conf_thresh,
                                                          nms_thresh=self.nms_thresh)

        if self.show_img:
            visualize(img, det_bboxes, det_cats, self.loader, scores=det_probs)

        for i in range(len(det_cats)):
            self.dets[det_cats[i]].append((img_id, det_probs[i]) + tuple(det_bboxes[i]))

    def eval_category(self, cid):
        cgt = self.gt[cid]
        cdets = np.array(self.dets[cid])
        if (cdets.shape == (0, )):
            return None, None
        scores = cdets[:, 1]
        sorted_inds = np.argsort(-scores)
        image_ids = cdets[sorted_inds, 0].astype(int)
        BB = cdets[sorted_inds]

        npos = 0
        for img_gt in cgt.values():
            img_gt['det'] = np.zeros(len(img_gt['difficult']), dtype=np.bool)
            npos += np.sum(~img_gt['difficult'])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            ovmax = -np.inf
            if image_ids[d] in cgt:
                R = cgt[image_ids[d]]
                bb = BB[d, 2:].astype(float)

                BBGT = R['bbox'].astype(float)

                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 0] + BBGT[:, 2], bb[0] + bb[2])
                iymax = np.minimum(BBGT[:, 1] + BBGT[:, 3], bb[1] + bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih

                # union
                uni = (bb[2] * bb[3] + BBGT[:, 2] * BBGT[:, 3] - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > 0.5:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = True
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
        return rec, prec


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):

            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class COCOEval(Evaluation):
    def __init__(self, net, loader, ckpt, conf_thresh=0.5, nms_thresh=0.3):
        super().__init__(net, loader, ckpt, conf_thresh, nms_thresh)
        self.filename = '/home/lear/kshmelko/scratch/coco_eval_{}.json'.format(args.run_name)

    def process_image(self, img_id):
        img, scale = self.loader.load_image(img_id)
        gt_bboxes, gt_cats = self.loader.read_annotations(img_id)[:2]
        proposals = self.loader.read_proposals(img_id)
        proposals = rescale_bboxes(proposals, scale)
        gt_bboxes = rescale_bboxes(gt_bboxes, scale)

        det_cats, det_probs, det_bboxes = self.net.detect(img, proposals,
                                                          conf_thresh=self.conf_thresh,
                                                          nms_thresh=self.nms_thresh)

        detections = []
        for j in range(len(det_cats)):
            obj = {}
            obj['bbox'] = list(map(float, det_bboxes[j]/scale))
            obj['score'] = float(det_probs[j])
            obj['image_id'] = img_id
            obj['category_id'] = self.loader.ids_to_coco_ids[det_cats[j]]
            detections.append(obj)
        return detections

    def compute_ap(self):
        coco_res = self.loader.coco.loadRes(self.filename)

        cocoEval = COCOeval(self.loader.coco, coco_res)
        cocoEval.params.imgIds = self.image_ids
        cocoEval.params.catIds = self.loader.included_coco_ids
        cocoEval.params.useSegm = False

        ev_res = cocoEval.evaluate()
        acc = cocoEval.accumulate()
        summarize = cocoEval.summarize()

    def evaluate_network(self, eval_first_n):
        detections = []

        start = 0
        cache = '%sEvalCache/%s_%i.pickle' % (self.loader.root, args.run_name,
                                              self.ckpt)
        if os.path.exists(cache):
            log.info("Found a partial eval cache: %s", cache)
            with open(cache, 'rb') as f:
                detections, start = pickle.load(f)

        bar = progressbar.ProgressBar()
        self.image_ids = list(sorted(self.loader.coco.getImgIds()))[:eval_first_n]
        for i in bar(range(start, len(self.image_ids))):
            img_id = self.image_ids[i]
            detections.extend(self.process_image(img_id))
            if i % 10 == 0 and i > 0:
                with open(cache, 'wb') as f:
                    pickle.dump((detections, i), f, pickle.HIGHEST_PROTOCOL)

        with open(self.filename, 'w') as f:
            json.dump(detections, f)
        self.compute_ap()


def visualize(image, bboxes, cat_ids, loader, color='blue', scores=None):
    fig = plt.figure(0)
    plt.cla()
    plt.clf()
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(cat_ids)):
        bbox = bboxes[i]
        cat = loader.ids_to_cats[cat_ids[i]]
        if scores is None:
            title = cat
        else:
            title = '{:s} {:.3f}'.format(cat, scores[i])
        ax.add_patch(plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            fill=False,
            edgecolor='red',
            linewidth=2))
        ax.text(bbox[0],
                bbox[1] - 2,
                title,
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=14,
                color='white')
    plt.show()
