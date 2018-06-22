import numpy as np
import logging

log = logging.getLogger()


# TODO reimplement on tf
# def filter_proposals(proposals, gt_bboxes):
#     if len(gt_bboxes) > 0:
#         idx = np.where(batch_iou(proposals, gt_bboxes).max(axis=1) < 0.5)[0]
#         proposals = proposals[idx]
#     return proposals, idx


# Taken from py-faster-rcnn
def nms_bbox(dets, scores, thresh=0.5):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rescale_bboxes(bboxes, scale):
    return np.int32(bboxes*scale)


# TODO include in tf?
def restore_bboxes(tcoords, anchors, width, height):
    assert tcoords.shape == anchors.shape
    t_x = tcoords[:, 0]*0.07
    t_y = tcoords[:, 1]*0.07
    t_w = tcoords[:, 2]*0.13
    t_h = tcoords[:, 3]*0.13
    a_w = anchors[:, 2]
    a_h = anchors[:, 3]
    a_x = anchors[:, 0]+a_w/2
    a_y = anchors[:, 1]+a_h/2
    x = t_x*a_w + a_x
    y = t_y*a_h + a_y
    w = np.exp(t_w)*a_w
    h = np.exp(t_h)*a_h

    x1 = np.maximum(0, x - w/2)
    y1 = np.maximum(0, y - h/2)
    x2 = np.minimum(width, w + x1)
    y2 = np.minimum(height, h + y1)
    return np.stack([x1, y1, x2-x1, y2-y1], axis=1)


def print_variables(name, var_list, level=logging.DEBUG):
    # TODO add more goodies:
    # - [ ] optional shape, dtype
    # - [ ] guess type: Operation or Tensor or Variable
    variables = sorted([v.op.name for v in var_list])
    s = "Variables to %s:\n%s" % (name, '\n'.join(variables))
    if level < 0:
        print(s)
    else:
        log.log(level, s)
