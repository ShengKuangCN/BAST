import os

"""Storage/working directory"""
DEFAULT_DIR = os.getcwd()


"""Raw data directory and converted data directory"""
DATA_DIR = DEFAULT_DIR + os.path.sep + 'data' + os.path.sep + 'data' + os.path.sep
CONVERTED_DATA_DIR = DEFAULT_DIR + os.path.sep + 'data' + os.path.sep + 'convert' + os.path.sep
MODEL_SAVE = DEFAULT_DIR + os.path.sep + 'ckpt' + os.path.sep + 'BAST' + os.path.sep
MODEL_NAME = 'BAST'
FIG_PATH = DEFAULT_DIR + os.path.sep + 'figure' + os.path.sep + 'BAST' + os.path.sep
EVAL_PATH = DEFAULT_DIR + os.path.sep + 'eval' + os.path.sep + 'BAST' + os.path.sep

"""GPU settings"""
GPU_LIST = [0]

"""Training set"""
DATA_ENV = 'RI'

"""Hyperparameters"""
SPECTROGRAM_SIZE = [129, 61]
PATCH_SIZE = 16
PATCH_OVERLAP = 10
NUM_OUTPUT = 2
EMBEDDING_DIM = 1024
TRANSFORMER_DEPTH = 3
TRANSFORMER_HEADS = 16
TRANSFORMER_MLP_DIM = 1024
TRANSFORMER_DIM_HEAD = 64
INPUT_CHANNEL = 1
DROPOUT = 0.2
EMB_DROPOUT = 0.2
LR = 0.0001
EPOCH = 20
BATCH_SIZE = 20
BATCH_SIZE = BATCH_SIZE * len(GPU_LIST)
POLAR_OUTPUT = False

BINAURAL_INTEGRATION = 'SUB'
LOSS = 'MIX'  # MIX MSE AD
POLAR_OUTPUT = False
SHARE_PARAMS = False
TRANSFORMER_POOL = 'mean'
if not SHARE_PARAMS:
    BATCH_SIZE -= 15
if BINAURAL_INTEGRATION == 'CONCAT':
    BATCH_SIZE -= 20

TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.25
