import argparse

parser = argparse.ArgumentParser(description='Train or eval a FastRCNN trained on VOC or COCO.')
parser.add_argument("--run_name", type=str, required=True, help='Name of the current run to properly store logs and checkpoints')
parser.add_argument("--ckpt", default=0, type=int, help='Resume training from this checkpoint. Use the most recent one if 0')

parser.add_argument("--dataset", default='voc07', choices=['voc07', 'voc12', 'coco'])
parser.add_argument("--proposals", default='', choices=['mcg', 'edgeboxes', ''], help='Which proposals to use? Empty string means default per-dataset choice (MCG for COCO, EdgeBoxes for VOC).')
parser.add_argument("--num_classes", required=True, type=int, help='Train on this number of classes (first N).')
parser.add_argument("--extend", default=0, type=int, help='Extend existing network by this number of classes incrementally and train on them.')
parser.add_argument("--num_layers", default=56, type=int, help='Number of ResNet layers')
parser.add_argument("--action", required=True, type=str, 'Comma-separated list of actions. Implemented actions: train, eval.')
parser.add_argument("--data_format", default='NHWC', choices=['NHWC', 'NCHW'], help='Data format for conv2d. Using of NCHW gives more cudnn acceleration')
parser.add_argument("--sigmoid", default=False, action='store_true', help='Use sigmoid instead of softmax on the last layer.')
parser.add_argument("--print_step", default=10, type=int, help='Print training logs every N iterations')

# EVALUATION OPTIONS
parser.add_argument("--conf_thresh", default=0.5, type=float, help='Threshold detections with this confidence level.')
parser.add_argument("--nms_thresh", default=0.3, type=float, help='Do NMS on FastFRCNN output with this IoU threshold')
parser.add_argument("--eval_first_n", default=1000000, type=int, help='Only evaluate on first N images from dataset. Useful for COCO, for example')
parser.add_argument("--eval_ckpts", default='', type=str, help='Comma-separated list of checkpoints to evaluate. Supports k as suffix for thousands.')

# TRAINING OPTIONS
parser.add_argument("--batch_size", default=64, type=int, help='Number of proposals per batch')
parser.add_argument("--num_images", default=2, type=int, help='Number of images per batch')
parser.add_argument("--num_positives_in_batch", default=16, type=int, help='Number of positive proposals in the batch.')
parser.add_argument("--pretrained_net", default='', type=str, help='Run name for network we use to extend incrementally')
parser.add_argument("--train_vars", default='', type=str, help='Comma-separated list of substrings. If variable name contains any of them, it is going to be trained. Empty list disables this filtering.')
parser.add_argument("--optimizer", default='nesterov', choices=['adam', 'nesterov', 'sgd', 'momentum'])
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--lr_decay", default=[], nargs='+', type=int, help='Space-separated list of steps where learning rate decays by factor of 10.')
parser.add_argument("--max_iterations", default=1000000, type=int, help='Total number of SGD steps.')
parser.add_argument("--reset_slots", default=True, type=bool, help='Should we clear out optimizer slots (momentum and Adam stuff) when we extend network?')

# DISTILLATION
# Lambda coefficients balancing each loss term
parser.add_argument("--frcnn_loss_coef", default=1.0, type=float)
parser.add_argument("--class_distillation_loss_coef", default=1.0, type=float)
parser.add_argument("--bbox_distillation_loss_coef", default=1.0, type=float)

parser.add_argument("--distillation", default=False, action='store_true', help='Boolean flag activating distillation')
# TODO make it default?
parser.add_argument("--bias_distillation", default=False, action='store_true', help='Boolean flag activating biased distillation. Requires --distillation flag to work.')
parser.add_argument("--crossentropy", default=False, action='store_true', help='Boolean flag to use crossentropy distillation instead of L2 distillation of logits')
parser.add_argument("--smooth_bbox_distillation", default=True, action='store_true', help='Boolean flag to use smooth L1 bounding box loss for distillation instead of just L2')

# Data loading and preprocessing threads.
parser.add_argument("--num_dataset_readers", default=2, type=int)
parser.add_argument("--num_prep_threads", default=4, type=int)

# deprecated flags, don't do anything in current version
parser.add_argument("--filter_proposals", default=False, action='store_true')
parser.add_argument("--prefetch_all", default=False, action='store_true')

args = parser.parse_args()

LOGS = './logs/'
CKPT_ROOT = './checkpoints/'


def get_logging_config(run):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': { 
            'standard': { 
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
            'short': { 
                'format': '[%(levelname)s]: %(message)s'
            },
        },
        'handlers': { 
            'default': { 
                'level': 'INFO',
                'formatter': 'short',
                'class': 'logging.StreamHandler',
            },
            'file': { 
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': LOGS+run+'.log'
            },
        },
        'loggers': { 
            '': { 
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
        } 
    }
