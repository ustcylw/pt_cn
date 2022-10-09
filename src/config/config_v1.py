import os, sys
import torch


class CONFIGS:
    PRE_SYMBOL = 'centernet'
    POST_SYMBOL = '000001'

    MODE = 'train'  # 'train'  'val'  'test'
    SEED = 7
    
    NET_STRUCETURE = 'resnet'  # 'resnet'  'mobilenet'

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f'[************]  PROJECT DIR: {ROOT_DIR}')
    if not os.path.exists(ROOT_DIR):
        raise ValueError(f'PROJECT NOT EXIST!!! {ROOT_DIR}')
    DATASET_DIRS = [
        f'/data/ylw/datasets/voc/VOC2007/JPEGImages'
    ]
    TRAIN_DATA_FILES = r'/data/ylw/code/pl_cn/data/2007_train.txt'
    VAL_DATA_FILES = r'/data/ylw/code/pl_cn/data/2007_val.txt'
    TEST_DATA_FILES = r'/data/ylw/code/pl_cn/data/2007_test.txt'
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'checkpoints/{PRE_SYMBOL}_{POST_SYMBOL}')
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    LOGS_DIR = os.path.join(ROOT_DIR, f'logs/{PRE_SYMBOL}_{POST_SYMBOL}')
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)


    # AVAIL_GPUS = min(1, torch.cuda.device_count())
    

    COLOR = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]
    ]

    VOC_CLASSES_PATH = 'model_data/voc_classes.txt'
    VOC_CLASS = [
        'aeroplane', 'bicycle', 'bird', 
        'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 
        'cow', 'diningtable', 'dog', 
        'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]
    VOC_CLS_NUM = 20
    
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3

    DATASET_MOSAIC = True
    DATASET_MOSAIC_PROB = 0.5
    DATASET_MIXUP = False
    DATASET_MIXUP_PROB = 0.5
    DATASET_SPECIAL_AUG_RATIO = 0.7

    # train
    TRAIN_INPUT_SHAPE = [512, 512]
    TRAIN_AUGMENT = True
    TRAIN_BATCH_SIZE = 12
    TRAIN_MULTI_SCALE_TRAIN = True
    # TRAIN_IOU_THRESHOLD_LOSS = 0.5
    TRAIN_START_EPOCHS = 300
    TRAIN_EPOCHS = 2
    TRAIN_SAVE_MODEL_INTERVAL = 2
    TRAIN_NUMBER_WORKERS = 4
    TRAIN_OPTIM_MOMENTUM = 0.9
    TRAIN_OPTIM_WEIGHT_DECAY = 5e-4
    TRAIN_OPTIM_TYPE = 'adam'
    TRAIN_LR_INIT = 1e-3 # 1e-5 # 1e-4  1e-3
    TRAIN_LR_END = 1e-6
    TRAIN_LR_DECAY_TYPE = 'cos'
    TRAIN_WARMUP_EPOCHS = 2  # or None
    TRAIN_PRETRAINED_MODEL_NAME = '206.pth'
    TRAIN_PRETRAINED_DIR = os.path.join(ROOT_DIR, f'checkpoints/{PRE_SYMBOL}_{POST_SYMBOL}')
    TRAIN_LOG_INTERVAL = 60
    TRAIN_ACC_INTERVAL = 3
    
    # test
    TEST_IMG_SIZE = 640
    TEST_BATCH_SIZE = 1
    TEST_NUMBER_WORKERS = 0
    TEST_CONF_THRESH = 0.01
    TEST_NMS_THRESH = 0.5
    TEST_SCORE_THRESH = 0.5
    TEST_MULTI_SCALE_TEST = False
    TEST_FLIP_TEST = False
    TEST_SAVE = True
    TEST_SAVE_DIR = os.path.join(ROOT_DIR, 'results')
    TEST_SHOW = True
    TEST_PRETRAINED_MODEL_NAME = os.path.join(os.path.join(ROOT_DIR, f'checkpoints/{PRE_SYMBOL}_{POST_SYMBOL}'), 'epoch=79-val_loss=16.26-other_metric=0.00.pth')  #  'last.ckpt'


    DECODE_CONF_TH = 0.3
    DECODE_NMS_TH = 0.5
