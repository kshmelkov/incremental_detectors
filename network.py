from utils import nms_bbox, restore_bboxes
from utils_tf import smooth_l1, tf_random_sample
from config import args

import logging

import numpy as np
import tensorflow as tf

import resnet

slim = tf.contrib.slim

log = logging.getLogger()
DISTILLATION_SCOPE = "distillation"


class Network(object):
    def __init__(self, image=None, rois=None, var_scope="", reuse=False,
                 num_classes=None, distillation=False, proposals=None):
        self.num_classes = num_classes or args.num_classes
        self.var_scope = var_scope
        self.reuse = reuse
        if image is None and rois is None:
            self.image_ph = tf.placeholder(tf.float32, [None, None, 3])
            self.rois_ph = tf.placeholder(tf.float32, [None, 4])
        else:
            self.image_ph = image
            self.rois_ph = rois
        self.proposals = proposals
        if distillation:
            self.create_distillation_subnet()
        with tf.variable_scope(self.var_scope, "network", reuse=reuse):
            self.inference()
        if distillation:
            self.logits, self.logits_for_distillation = tf.split(self.logits, 2, axis=0)
            self.bboxes, self.bboxes_for_distillation = tf.split(self.bboxes, 2, axis=0)

    def compute_frcnn_crossentropy_loss(self, skipped_classes=0):
        # TODO think again about skipped classes in the light of cifar10 exps
        if args.sigmoid:
            oh_labels = tf.one_hot(self.cats, self.num_classes+1)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=oh_labels)
            cross_entropy = tf.reduce_sum(cross_entropy[:, (1+skipped_classes):], 1)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.cats)
        cross_entropy_mean = args.frcnn_loss_coef*tf.reduce_mean(cross_entropy)
        return cross_entropy_mean

    def compute_frcnn_bbox_loss(self):
        oh_labels = tf.one_hot(self.cats, self.num_classes+1)
        bboxes = tf.reshape(self.bboxes, (-1, self.num_classes+1, 4))
        gt_bboxes = tf.reshape(self.refine, (-1, 1, 4))
        bg_iverson = tf.to_float(tf.greater(self.cats, 0))
        bbox_loss = args.frcnn_loss_coef*tf.reduce_mean(bg_iverson*(
            tf.reduce_sum(oh_labels*smooth_l1(bboxes, gt_bboxes), axis=1)))
        return bbox_loss

    def compute_distillation_crossentropy_loss(self):
        cached_classes = self.num_classes - self.subnet.num_classes
        logits = self.logits_for_distillation[:, :cached_classes+1]

        if args.crossentropy and args.sigmoid:
            class_distillation_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[:, 1:],
                                                        labels=tf.sigmoid(self.distillated_logits[:, 1:])),
                axis=1), axis=0)
        if args.crossentropy and not args.sigmoid:
            cache_softmax = tf.nn.softmax(self.distillated_logits)
            pad_size = tf.shape(self.logits_for_distillation)[1] - tf.shape(self.distillated_logits)[1]
            padded_cache_softmax = tf.concat([cache_softmax,
                                              tf.zeros(tf.stack([tf.shape(self.distillated_logits)[0],
                                                                 pad_size]))], 1)
            class_distillation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits_for_distillation, labels=padded_cache_softmax))
        if not args.crossentropy and args.sigmoid:
            class_distillation_loss = tf.reduce_mean(
                tf.square(logits[:, 1:] - self.distillated_logits[:, 1:]))
        if not args.crossentropy and not args.sigmoid:
            logits = logits - tf.reduce_mean(self.logits_for_distillation,
                                             axis=1, keep_dims=True)
            distillated_logits = (self.distillated_logits -
                                  tf.reduce_mean(self.distillated_logits, axis=1, keep_dims=True))
            class_distillation_loss = tf.reduce_mean(tf.square(logits - distillated_logits))
        class_distillation_loss *= args.class_distillation_loss_coef

        return class_distillation_loss

    def compute_distillation_bbox_loss(self):
        cached_classes = self.num_classes - self.subnet.num_classes
        bboxes_full = tf.reshape(self.bboxes_for_distillation, (-1, self.num_classes+1, 4))
        bboxes = bboxes_full[:, 1:cached_classes+1, :]
        cache_bboxes = tf.reshape(self.distillated_bboxes, (-1, cached_classes+1, 4))[:, 1:, :]
        if args.smooth_bbox_distillation:
            bbox_distillation = smooth_l1(bboxes, cache_bboxes)
        else:
            bbox_distillation = tf.square(bboxes - cache_bboxes)
        bboxes_distillation_loss = args.bbox_distillation_loss_coef * tf.reduce_mean(bbox_distillation)
        return bboxes_distillation_loss

    def create_distillation_subnet(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm],
                            trainable=False):
            self.subnet = Network(self.image_ph, self.proposals,
                                  var_scope=DISTILLATION_SCOPE,
                                  reuse=self.reuse, num_classes=args.num_classes,
                                  distillation=False, proposals=self.proposals)
            logits = tf.stop_gradient(self.subnet.logits)
            bboxes = tf.stop_gradient(self.subnet.bboxes)

            assert args.bias_distillation  # XXX
            if args.bias_distillation:
                if args.filter_proposals:
                    raise NotImplemented
                #     all_proposals, idx = filter_proposals(proposals, gt_bboxes)
                #     if len(gt_bboxes) > 0:
                #         logits = logits[idx]
                #         bboxes = bboxes[idx]
                bg_prob = tf.nn.softmax(logits)[:, 0]
                _, sorted_idx = tf.nn.top_k(-bg_prob, 2*args.batch_size)
                dist_idx = tf_random_sample(args.batch_size, sorted_idx)[0]
                # TODO more stop grad?
                self.distillated_logits = tf.gather(logits, dist_idx)
                self.distillated_proposals = tf.gather(self.proposals, dist_idx)
                self.distillated_bboxes = tf.gather(bboxes, dist_idx)
                self.rois_ph = tf.concat([self.rois_ph, self.distillated_proposals], 0)

    def compute_train_accuracy(self):
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.cats, 1),
                                      "float"), name='train_acc')

    def compute_background_frequency(self):
        return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.logits, 1), 0)), name='bg_freq')

    def inference(self):
        total_classes = 20  # FIXME
        net, endpoints = resnet.create_trunk(tf.expand_dims(self.image_ph, 0),
                                             self.rois_ph, reuse=self.reuse,
                                             weight_decay=args.weight_decay)
        self.fc_out = endpoints['pooled']

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(args.weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            reuse=self.reuse):
            self.logits_layer = slim.fully_connected(self.fc_out, total_classes+1, scope='fc8',
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.logits = self.logits_layer[:, :self.num_classes+1]
            self.bboxes_layer = slim.fully_connected(self.fc_out, (total_classes+1)*4, scope='bboxes',
                                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
            self.bboxes = self.bboxes_layer[:, :(4*(self.num_classes+1))]

            self.softmax = tf.nn.softmax(self.logits)
            self.sigmoid = tf.sigmoid(self.logits)

    def _forward_pass(self, image, proposals, split=2):
        H, W, _ = image.shape
        proposals = proposals.astype(np.float32)
        # TODO make it a bit more consistent idk
        rois = np.array([(bbox[1]/H, bbox[0]/W, (bbox[1]+bbox[3])/H,
                          (bbox[0]+bbox[2])/W) for bbox in proposals])
        softmax, sigmoid, bboxes, logits = tf.get_default_session().run(
            [self.softmax, self.sigmoid,
             self.bboxes, self.logits],
            feed_dict={self.image_ph: image,
                       self.rois_ph: rois})
        return softmax, sigmoid, bboxes, logits

    def detect(self, image, proposals, conf_thresh=0.8, nms_thresh=0.3):
        height, width, _ = image.shape
        softmax, sigmoid, bboxes, logits = self._forward_pass(image, proposals)

        det_cats = []
        det_bboxes = []
        det_probs = []
        num_cats = softmax.shape[1]
        if args.sigmoid:
            scores = sigmoid
        else:
            scores = softmax

        for cid in range(1, num_cats):
            good = scores[:, cid] >= conf_thresh
            if not np.any(good):
                continue
            bboxes = bboxes.reshape((-1, num_cats, 4))
            cat_refines = bboxes[good, cid]
            cat_props = proposals[good]
            corrected_bboxes = restore_bboxes(cat_refines, cat_props, width, height)
            cat_probs = scores[good, cid]

            keep = nms_bbox(corrected_bboxes, cat_probs, nms_thresh)
            corrected_bboxes = corrected_bboxes[keep]
            cat_probs_ref = cat_probs[keep]
            for i in range(len(keep)):
                det_cats.append(cid)
                det_probs.append(cat_probs_ref[i])
                det_bboxes.append(corrected_bboxes[i])
        return det_cats, det_probs, det_bboxes
