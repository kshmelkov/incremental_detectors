# Incremental Learning of Object Detectors without Catastrophic Forgetting

This is code release for our paper ["Incremental Learning of Object Detectors without Catastrophic Forgetting"]( https://arxiv.org/abs/1708.06977) published on ICCV 2017.

## Requirements

Code is written for Python 3.5 and TensorFlow 1.5 (might require minor modifications for more recent versions). You are also expected to have normal scientific stack installed: NumPy, SciPy, Matplotlib, OpenCV.
If you don't like OpenCV, you can replace it with something that can read and resize images. SciPy is used only for interaction with Matlab.

You also need a checkpoint of [pre-trained ResNet-50](https://drive.google.com/drive/folders/1Xxs6jK_adXdr1asyyqxJiV3a3yUU2G9N?usp=sharing) to initialize an object detector. Put it in the directory `./resnet`. These weights are obtained from official Microsoft release, but slightly changed to correspond better to TF minor differences. This checkpoint is different from the one released by Google for TF-Slim.

## Datasets

All experiments were done on [PASCAL VOC 2007]*(http://host.robots.ox.ac.uk/pascal/VOC/*) and [Microsoft COCO]*(http://cocodataset.org/).
To use COCO you also need [pycocotools]*(https://github.com/cocodataset/cocoapi) installed.

## Experiments

To train and evaluate a normal FastRCNN on VOC 2007 launch the following command:

```
python3 frcnn.py sigmoid --run_name=resnet_sigmoid_20 --num_classes=20 --dataset=voc07 --max_iterations=40000 --action=train,eval --eval_first_n=5000 --eval_ckpts=40k --learning_rate=0.001 --sigmoid
```

To train 10 classes network and then extend it for 10 more classes:

```
python3 frcnn.py sigmoid --run_name=resnet_sigmoid_10 --num_classes=10 --dataset=voc07 --max_iterations=40000 --action=train,eval --eval_ckpts=40k --learning_rate=0.001 --lr_decay 30000 --sigmoid
python3 frcnn.py sigmoid --run_name=resnet_sigmoid_10_ext10 --num_classes=10 --extend=10 --dataset=voc07 --max_iterations=40000 --action=train,eval --eval_ckpts=40k --learning_rate=0.0001 --sigmoid --pretrained_net=resnet_sigmoid_10 --distillation --bias_distillation
```

The same way to train a COCO model on all classes:

```
python3 frcnn.py --run_name=resnet_coco_80 --num_classes=80 --dataset=coco --max_iterations=500000 --lr_decay_step=250000 --weight_decay=0.00005 --eval_first_n=5000 --eval_ckpts=500000 --action=train,eval --sigmoid"
```
