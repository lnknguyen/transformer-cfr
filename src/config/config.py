from yacs.config import CfgNode as CN
from datetime import datetime
import os

# YACS overwrite these settings using YAML
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type
_C.MODEL.NAME = "trans"

_C.MODEL.EMBEDDING_DIM = 300
_C.MODEL.DROPOUT_P = 0.0
_C.MODEL.ENCODER_NUM_LAYER = 2

_C.MODEL.DIAG_INPUT_VOCAB_SIZE = 1000
_C.MODEL.PROC_INPUT_VOCAB_SIZE = 1000
_C.MODEL.VOCAB_SIZE = 258

_C.MODEL.N_OUT_CLASSES = 2
_C.MODEL.FC_HIDDEN_SIZE = 16


_C.MODEL.MAX_SEQ_LENGTH = 200

# Transformer Params
_C.MODEL.NUM_HEADS = 8
_C.MODEL.TRANS_DEPTH = 2

# ---------------------------------------------------------------------------- #
# TRAIN options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.EPOCHS = 20
_C.TRAIN.LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BATCH_SIZE = 16

# ---------------------------------------------------------------------------- #
# Path options
# ---------------------------------------------------------------------------- #
_C.PATH = CN()
_C.PATH.ROOT = "results/" + _C.MODEL.NAME
_C.PATH.HOME_DIR = "../"
_C.PATH.MODEL_OUT_DIR = _C.PATH.HOME_DIR + _C.PATH.ROOT

# ---------------------------------------------------------------------------- #
# Utils options
# ---------------------------------------------------------------------------- #
_C.UTILS = CN()

_C.UTILS.TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")