#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt

from config import args, get_logging_config, CKPT_ROOT

from tabulate import tabulate
import pickle

from voc_loader import VOCLoader, VOC_CATS, create_permutation
from coco_loader import COCOLoader
from network import Network, DISTILLATION_SCOPE
from utils import print_variables
from utils_tf import yxyx_to_xywh, preprocess_proposals, mirror_distortions, xywh_to_yxyx
from evaluation import Evaluation, AVERAGE, COCOEval
import datasets
import resnet

from datetime import datetime

import progressbar
import time
import os
import sys
import subprocess
import re
import socket

import logging
import logging.config

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()

train_dir = CKPT_ROOT + args.run_name
pretrain_dir = CKPT_ROOT + args.pretrained_net


def extract_batch(data_provider, classes):
    with tf.device("/cpu:0"):
        im, bbox, gt, proposals = data_provider.get(['image', 'object/bbox', 'object/label', 'proposal/bbox'])
        im = tf.to_float(im)/255
        gt = tf.to_int32(gt)
        gt_mask = tf.reduce_any(tf.equal(tf.expand_dims(gt, 1), tf.expand_dims(tf.constant(classes), 0)), axis=1)
        gt = tf.boolean_mask(gt, gt_mask)
        bbox = tf.boolean_mask(bbox, gt_mask)

        sh = tf.to_float(tf.shape(im))
        h, w = sh[0], sh[1]
        scale = tf.minimum(1000.0/tf.maximum(h, w), 600.0/tf.minimum(h, w))
        new_dims = tf.to_int32((h*scale, w*scale))
        im = tf.image.resize_images(im, new_dims)
        bbox = yxyx_to_xywh(tf.clip_by_value(bbox, 0.0, 1.0))
        proposals = yxyx_to_xywh(tf.clip_by_value(proposals, 0.0, 1.0))

        num_gt = tf.shape(bbox)[0]
        rois = tf.concat([bbox, proposals], 0)
        im, rois = mirror_distortions(im, rois)
        bbox = rois[:num_gt]
        proposals = rois[num_gt:]

        # TODO stop gradient somewhere?
        batch_prop, batch_gt, batch_refine, include = preprocess_proposals(proposals, bbox, gt)

        return tf.train.maybe_batch([im, batch_prop, batch_refine, batch_gt, xywh_to_yxyx(proposals)],
                                    include, 1, capacity=128, num_threads=args.num_prep_threads,
                                    dynamic_pad=True)


def restore_ckpt(ckpt_dir=None, global_step=None, ckpt_num=0):
    ckpt_dir = ckpt_dir or (CKPT_ROOT+args.run_name)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_num = ckpt_num or args.ckpt
        if ckpt_num > 0:
            ckpt_to_restore = ckpt_dir+'/model.ckpt-%i' % ckpt_num
        else:
            ckpt_to_restore = ckpt.model_checkpoint_path
        if args.reset_slots:
            variables_to_restore = slim.get_model_variables()
            if global_step is not None:
                variables_to_restore += [global_step]
        else:
            variables_to_restore = tf.global_variables()
        variables_to_restore = [v for v in variables_to_restore
                                if 'distillation' not in v.op.name]
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            ckpt_to_restore, variables_to_restore)
        log.info("Restore from %s", ckpt_to_restore)
    else:
        init_assign_op = tf.no_op()
        init_feed_dict = None
        log.info("Completely new network")

    return init_assign_op, init_feed_dict


def get_total_loss(networks):
    frcnn_xe_loss = tf.reduce_mean([net.compute_frcnn_crossentropy_loss() for net in networks])
    tf.summary.scalar('loss/frcnn/class', frcnn_xe_loss)
    frcnn_bbox_loss = tf.reduce_mean([net.compute_frcnn_bbox_loss() for net in networks])
    tf.summary.scalar('loss/frcnn/bbox', frcnn_bbox_loss)
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('loss/weight_decay', l2_loss)
    total_loss = frcnn_xe_loss + frcnn_bbox_loss + l2_loss

    if args.distillation:
        distillation_xe_loss = tf.reduce_mean([net.compute_distillation_crossentropy_loss() for net in networks])
        tf.summary.scalar('loss/distillation/class', distillation_xe_loss)
        distillation_bbox_loss = tf.reduce_mean([net.compute_distillation_bbox_loss() for net in networks])
        tf.summary.scalar('loss/distillation/bbox', distillation_bbox_loss)
        total_loss += distillation_xe_loss + distillation_bbox_loss
    tf.summary.scalar('loss/total', total_loss)
    return total_loss


def train_network(sess):
    to_learn, prefetch_classes, remain = split_classes()
    # TODO if args.prefetch_all
    # if args.prefetch_all:
    #     prefetch_classes = sorted(list(set(to_learn_classes+prefetch_classes)))

    ### data loading ###
    with tf.device("/cpu:0"):
        dataset = datasets.get_dataset('voc07-trainval-proposals')
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, num_readers=args.num_dataset_readers,
            common_queue_capacity=512, common_queue_min=32)

    ### network graph construction ###
    networks = []
    num_classes = args.num_classes + args.extend
    for i in range(args.num_images):
        with tf.name_scope('img%i' % i):
            dequeue = extract_batch(data_provider, prefetch_classes)
            image, rois, refine, cats, proposals = [t[0] for t in dequeue]
            net = Network(image=image, rois=rois, reuse=(i > 0),
                          num_classes=num_classes,
                          distillation=args.distillation, proposals=proposals)
            net.refine = refine
            net.cats = cats
            networks.append(net)

    ### launching queues
    coord = tf.train.Coordinator()
    log.debug("Launching prefetch threads")
    prefetch_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    close_all_queues = tf.group(*[qr.close_op for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)])

    ### metrics ###
    train_acc = tf.reduce_mean([net.compute_train_accuracy() for net in networks])
    tf.summary.scalar('accuracy/train', train_acc)
    bg_freq = tf.reduce_mean([net.compute_background_frequency() for net in networks])
    tf.summary.scalar('bg_freq/train', bg_freq)

    ### setup training ###
    train_vars = [v for v in tf.trainable_variables()
                  if v.op.name.endswith('weights') or v.op.name.endswith('biases')]
    if args.train_vars != '':
        var_substrings = args.train_vars.split(',')
        train_vars = [v for v in train_vars
                      if np.any([s in v.op.name for s in var_substrings])]
    print_variables('train', train_vars)
    total_loss = get_total_loss(networks)
    global_step = slim.get_or_create_global_step()
    opt = get_optimizer(global_step)
    train_op = slim.learning.create_train_op(
        total_loss, opt,
        global_step=global_step,
        variables_to_train=train_vars,
        summarize_gradients=True)

    ### summaries
    slim.summarize_variables()
    partial_summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss'))
    summary_op = tf.summary.merge_all()

    ### create all initializers or checkpoint assignments
    clean_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    imagenet_init_op, imagenet_feed_dict = resnet.get_imagenet_init()
    has_pretrained = args.extend != 0 and args.pretrained_net != ''
    if has_pretrained:
        preinit_assign_op, preinit_feed_dict = restore_ckpt(ckpt_dir=CKPT_ROOT+args.pretrained_net)
    init_assign_op, init_feed_dict = restore_ckpt(ckpt_dir=train_dir, global_step=global_step)
    if args.distillation:
        dist_init_op, dist_init_feed_dict = init_dist_network()

    ### final preparations for training
    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=30)
    tf.get_default_graph().finalize()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    ### run variable restore or init
    sess.run(clean_init_op)
    sess.run(imagenet_init_op, feed_dict=imagenet_feed_dict)
    if has_pretrained:
        sess.run(preinit_assign_op, feed_dict=preinit_feed_dict)
    sess.run(init_assign_op, feed_dict=init_feed_dict)
    if args.distillation:
        sess.run(dist_init_op, feed_dict=dist_init_feed_dict)

    ### train loop
    starting_step = sess.run(global_step)
    log.info("Starting training...")
    for step in range(starting_step, args.max_iterations+1):
        start_time = time.time()
        try:
            train_loss, acc, bg_freq_iter, summary_str = sess.run([train_op, train_acc, bg_freq, partial_summary_op])
        except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
            break
        except KeyboardInterrupt:
            log.info("Killed by ^C")
            break
        duration = time.time() - start_time

        summary_writer.add_summary(summary_str, step)

        num_examples_per_step = len(networks)
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        if step % args.print_step == 0:
            format_str = ('step %d, loss = %.2f, acc = %.2f bg = %.2f (%.1f ex/sec; %.3f '
                          'sec/batch)')
            log.info(format_str % (step, train_loss, acc, bg_freq_iter,
                                   examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0 and step > 0:
            summary_writer.flush()
            log.debug("Saving checkpoint...")
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

    summary_writer.close()
    coord.request_stop()
    try:
        sess.run(close_all_queues)
    except tf.errors.CancelledError:
        # silently skip because at this point it is useless
        pass
    coord.join(prefetch_threads)


def get_optimizer(global_step):
    learning_rate = args.learning_rate

    if len(args.lr_decay) > 0:
        steps = []
        learning_rates = [learning_rate]
        for i, step in enumerate(args.lr_decay):
            steps.append(step)
            learning_rates.append(learning_rate*10**(-i-1))
        learning_rate = tf.train.piecewise_constant(tf.to_int32(global_step),
                                                    steps, learning_rates)
    tf.summary.scalar('learning_rate', learning_rate)

    if args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif args.optimizer == 'nesterov':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    elif args.optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=False)
    elif args.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError
    return opt


# TODO refactor it inside get_loader
def split_classes():
    num_classes = args.num_classes
    if args.extend != 0:
        num_classes += args.extend
    # FIXME loader!
    if args.dataset == 'coco':
        total_number = 80
    else:
        total_number = 20
    original = list(range(total_number+1))
    to_learn = list(range(num_classes+1))
    remaining = [i for i in original if i not in to_learn]
    if args.extend != 0:
        prefetch_cats = args.extend
        prefetch_cats = to_learn[-prefetch_cats:]
    else:
        prefetch_cats = to_learn
    log.debug('Splitting classes into %s, %s, %s', to_learn, prefetch_cats, remaining)
    return to_learn, prefetch_cats, remaining


def init_dist_network():
    for v in tf.global_variables():
        print(v.op.name)
    ckpt_dir = CKPT_ROOT+args.pretrained_net
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        variables_to_restore = slim.get_model_variables(scope=DISTILLATION_SCOPE)
        var_dict = {v.op.name[len(DISTILLATION_SCOPE)+1:]: v for v in variables_to_restore}
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            ckpt.model_checkpoint_path, var_dict)
        print("Restoring %s" % ckpt.model_checkpoint_path)
    else:
        raise ValueError("Pretrained network not found: %s" % ckpt_dir)
    return init_assign_op, init_feed_dict


def eval_network(sess):
    net = Network(num_classes=args.num_classes+args.extend, distillation=False)
    _, _, remain = split_classes()
    loader = get_loader(False, remain)
    is_voc = loader.dataset == 'voc'

    if args.eval_ckpts != '':
        ckpts = args.eval_ckpts.split(',')
    else:
        ckpts = [args.ckpt]
    global_results = {cat: [] for cat in loader.categories}
    global_results[AVERAGE+" 1-10"] = []
    global_results[AVERAGE+" 11-20"] = []
    global_results[AVERAGE+" ALL"] = []
    for ckpt in ckpts:
        if ckpt[-1].lower() == 'k':
            ckpt_num = int(ckpt[:-1])*1000
        else:
            ckpt_num = int(ckpt)
        init_op, init_feed_dict = restore_ckpt(ckpt_num=ckpt_num)
        sess.run(init_op, feed_dict=init_feed_dict)
        log.info("Checkpoint {}".format(ckpt))
        if is_voc:
            results = Evaluation(net, loader, ckpt_num, args.conf_thresh, args.nms_thresh).evaluate_network(args.eval_first_n)
            for cat in loader.categories:
                global_results[cat].append(results[cat] if cat in results else 0.0)
            # TODO add output formating, line after learnt cats
            old_classes = [results.get(k, 0) for k in loader.categories[:10]]
            new_classes = [results.get(k, 0) for k in loader.categories[10:]]
            all_classes = [results.get(k, 0) for k in loader.categories]
            global_results[AVERAGE+" 1-10"].append(np.mean(old_classes))
            global_results[AVERAGE+" 11-20"].append(np.mean(new_classes))
            global_results[AVERAGE+" ALL"].append(np.mean(all_classes))

            headers = ['Category'] + [("mAP (%s, %i img)" % (ckpt, args.eval_first_n)) for ckpt in ckpts]
            table_src = []
            for cat in loader.categories:
                table_src.append([cat] + global_results[cat])
            table_src.append([AVERAGE+" 1-10", ] + global_results[AVERAGE+" 1-10"])
            table_src.append([AVERAGE+" 11-20", ] + global_results[AVERAGE+" 11-20"])
            table_src.append([AVERAGE+" ALL", ] + global_results[AVERAGE+" ALL"])
            out = tabulate(table_src, headers=headers,
                           floatfmt=".1f", tablefmt='orgtbl')
            with open("/home/lear/kshmelko/scratch/logs/results_voc/%s.pkl" % args.run_name, 'wb') as f:
                pickle.dump(global_results, f, pickle.HIGHEST_PROTOCOL)
            log.info("Summary table over %i checkpoints\nExperiment: %s\n%s", len(ckpts), args.run_name, out)
        else:
            results = COCOEval(net, loader, ckpt_num, args.conf_thresh, args.nms_thresh).evaluate_network(args.eval_first_n)


def look_ckpt(ckpt_dir, ckpt_num, fail_if_absent=False):
    # TODO support for k
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if ckpt_num == 0:
            ckpt_to_restore = ckpt.model_checkpoint_path
        else:
            ckpt_to_restore = ckpt_dir+'/model.ckpt-%i' % ckpt_num
        log.info("Restoring model %s..." % ckpt_to_restore)
        return ckpt_to_restore
    else:
        log.warning("No checkpoint to restore in {}".format(ckpt_dir))
        if fail_if_absent:
            quit(2)
        else:
            return None


def get_loader(is_training, excluded=[]):
    if args.dataset == 'coco':
        loader = COCOLoader
        year = '2014'
        split = 'train' if is_training else 'val'
        proposals = args.proposals or 'mcg'
    elif args.dataset == 'voc07':
        loader = VOCLoader
        year = '07'
        split = 'trainval' if is_training else 'test'
        proposals = args.proposals or 'edgeboxes'
    elif args.dataset == 'voc12':
        loader = VOCLoader
        year = '12'
        split = 'train' if is_training else 'val'
        proposals = args.proposals or 'edgeboxes'
    else:
        raise NotImplementedError

    return loader(year, proposals, split, excluded=excluded)


def main(argv=None):  # pylint: disable=unused-argument
    for action in args.action.split(','):
        assert action in ['train', 'eval']
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
            if action == 'train':
                train_network(sess)
            if action == 'eval':
                eval_network(sess)
    log.info("End of script: %s", ' '.join(sys.argv))


if __name__ == '__main__':
    exec_string = ' '.join(sys.argv)
    log.debug("Executing a command: %s", exec_string)
    cur_commit = subprocess.check_output("git log -n 1 --pretty=format:\"%H\"".split())
    cur_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
    git_diff = subprocess.check_output('git diff --no-color'.split()).decode('ascii')
    log.debug("on branch %s with the following diff from HEAD (%s):" % (cur_branch, cur_commit))
    log.debug(git_diff)
    hostname = socket.gethostname()
    if 'gpuhost' in hostname:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        nvidiasmi = subprocess.check_output('nvidia-smi').decode('ascii')
        log.debug("Currently we are on %s and use gpu%s:" % (hostname, gpu_id))
        log.debug(nvidiasmi)
    tf.app.run()
