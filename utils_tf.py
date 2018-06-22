import tensorflow as tf
from config import args


def smooth_l1(x, y):
    abs_diff = tf.abs(x-y)
    return tf.reduce_sum(tf.where(abs_diff < 1, 0.5*abs_diff*abs_diff, abs_diff - 0.5), reduction_indices=-1)


def batch_iou(proposals, gt):
    bboxes = tf.reshape(tf.transpose(proposals), [4, -1, 1])
    bboxes_x1 = bboxes[0]
    bboxes_x2 = bboxes[0]+bboxes[2]
    bboxes_y1 = bboxes[1]
    bboxes_y2 = bboxes[1]+bboxes[3]

    gt = tf.reshape(tf.transpose(gt), [4, 1, -1])
    gt_x1 = gt[0]
    gt_x2 = gt[0]+gt[2]
    gt_y1 = gt[1]
    gt_y2 = gt[1]+gt[3]

    widths = tf.maximum(0.0, tf.minimum(bboxes_x2, gt_x2) -
                        tf.maximum(bboxes_x1, gt_x1))
    heights = tf.maximum(0.0, tf.minimum(bboxes_y2, gt_y2) -
                         tf.maximum(bboxes_y1, gt_y1))
    intersection = widths*heights
    union = bboxes[2]*bboxes[3] + gt[2]*gt[3] - intersection
    return (intersection / union)


def xywh_to_yxyx(xywh):
    x, y, w, h = tf.unstack(xywh, axis=1)
    return tf.stack([y, x, y+h, x+w], axis=1)


def yxyx_to_xywh(yxyx):
    y1, x1, y2, x2 = tf.unstack(yxyx, axis=1)
    return tf.stack([x1, y1, x2-x1, y2-y1], axis=1)


def encode_bboxes_tf(proposals, gt):
    prop_x = proposals[:, 0]
    prop_y = proposals[:, 1]
    prop_w = proposals[:, 2]
    prop_h = proposals[:, 3]

    gt_x = gt[:, 0]
    gt_y = gt[:, 1]
    gt_w = gt[:, 2]
    gt_h = gt[:, 3]

    diff_x = (gt_x + 0.5*gt_w - prop_x - 0.5*prop_w)/prop_w
    diff_y = (gt_y + 0.5*gt_h - prop_y - 0.5*prop_h)/prop_h
    diff_w = tf.log(gt_w/prop_w)
    diff_h = tf.log(gt_h/prop_h)

    # TODO extern std values from here
    x = tf.stack([diff_x/0.07, diff_y/0.07, diff_w/0.13, diff_h/0.13], axis=1)
    return x


def mirror_distortions(image, rois, prob=0.5):
    x, y, w, h = tf.unstack(rois, axis=1)
    flipped_rois = tf.stack([1.0 - x - w, y, w, h], axis=1)
    return tf.cond(tf.random_uniform([], 0, 1.0) < prob,
                   lambda: (tf.image.flip_left_right(image), flipped_rois),
                   lambda: (image, rois))


def tf_random_sample(sz, *args):
    s = tf.reshape((tf.shape(args[0])[0]), (1, ))
    ar = tf.expand_dims(tf.log(tf.tile([10.], s)), 0)
    sample = tf.multinomial(ar, sz)[0]
    return tuple(tf.gather(a, sample) for a in args)


def filter_small_gt(gt_bboxes, gt_cats, min_size):
    mask = tf.logical_and(gt_bboxes[:, 2] >= min_size,
                          gt_bboxes[:, 3] >= min_size)
    return tf.boolean_mask(gt_bboxes, mask), tf.boolean_mask(gt_cats, mask)


def preprocess_proposals(proposals, gt_bboxes, gt_cats):
    # this ugly bunch of functions is needed to cleanly use tf.cond
    # in order to avoid zero-length array reductions that are all over
    # the place in FRCNN batch preparation
    def empty_batch():
        return [tf.zeros((args.batch_size, 4), dtype=tf.float32),
                tf.zeros((args.batch_size,), dtype=tf.int32),
                tf.zeros((args.batch_size, 4), dtype=tf.float32),
                tf.zeros((), dtype=tf.bool)]

    def sample_boxes(proposals, good_proposals_mask, iou_matrix, gt_cats):
        proposals_cats = tf.argmax(tf.boolean_mask(iou_matrix, good_proposals_mask), axis=1)
        bad_proposals_mask = tf.logical_not(good_proposals_mask)

        pos_cats = tf.gather(gt_cats, proposals_cats)
        positive_proposals = tf.boolean_mask(proposals, good_proposals_mask)
        negative_proposals = tf.boolean_mask(proposals, bad_proposals_mask)

        refine = encode_bboxes_tf(positive_proposals, tf.gather(gt_bboxes, proposals_cats))
        positive_proposals, pos_cats, pos_refine = tf_random_sample(args.num_positives_in_batch,
                                                                    positive_proposals,
                                                                    pos_cats, refine)
        num_negatives = args.batch_size - args.num_positives_in_batch
        negative_proposals = tf_random_sample(num_negatives,
                                              negative_proposals)[0]
        proposals = tf.concat([positive_proposals, negative_proposals], 0)

        neg_cats = tf.zeros((num_negatives, ), dtype=tf.int32)
        cats = tf.concat([pos_cats, neg_cats], 0)
        neg_refine = tf.zeros((num_negatives, 4), dtype=tf.float32)
        refine = tf.concat([pos_refine, neg_refine], 0)

        return [xywh_to_yxyx(proposals), cats, refine, tf.ones((), dtype=tf.bool)]

    def prepare_batch(proposals, gt_bboxes, gt_cats, iou_threshold=0.5):
        # bboxes are expected in xywh format [0; 1]
        iou_matrix = batch_iou(proposals, gt_bboxes)  # shape = (n_proposals, n_gt)
        overlap = tf.reduce_max(iou_matrix, axis=1)
        good_proposals_mask = overlap >= iou_threshold
        # TODO < fixed number?
        any_positive = tf.reduce_sum(tf.to_int32(good_proposals_mask)) > 4
        return tf.cond(any_positive,
                       lambda: sample_boxes(proposals, good_proposals_mask, iou_matrix, gt_cats),
                       empty_batch)

    empty_gt = tf.equal(tf.cast(tf.shape(gt_cats)[0], tf.int32), 0)
    return tf.cond(empty_gt, empty_batch, lambda: prepare_batch(proposals, gt_bboxes, gt_cats))
