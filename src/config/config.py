from yacs.config import CfgNode as CN
from datetime import datetime
import os

# YACS overwrite these settings using YAML
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ---------------------------------------------------------------------------- #
# Simulation options
# ---------------------------------------------------------------------------- #
_C.SIM = CN()

# Noise level for input
_C.SIM.INPUT_NOISE = 0.2

# Noise level for input
_C.SIM.OUTPUT_NOISE = 0.2

# Bias level
_C.SIM.BIAS_RATE = 0.2

_C.SIM.TREATMENT_STRENGTH = 0.2
_C.SIM.CONFOUNDER_STRENGTH = 0.2

# Potential output type. Possible values: binary, continuous
_C.SIM.OUTPUT_TYPE = 'binary'
# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type
_C.MODEL.NAME = "trans"
_C.MODEL.EXP_NAME = "exp0"
_C.MODEL.EMBEDDING_DIM = 300
_C.MODEL.DROPOUT_P = 0.0
_C.MODEL.ENCODER_NUM_LAYER = 2

# Number of unique diagnosis + procedure codes
_C.MODEL.DIAG_VOCAB_SIZE = 3448
_C.MODEL.VOCAB_SIZE = 258

_C.MODEL.N_OUT_CLASSES = 2
_C.MODEL.FC_HIDDEN_SIZE = 16

# Max codes given in a single visit
_C.MODEL.MAX_SEQ_LENGTH = 66

# Max visits made by a single patient
_C.MODEL.MAX_VISIT_LENGTH = 20

# Transformer Params
_C.MODEL.ATTENTION_HEADS = 8
_C.MODEL.TRANS_DEPTH = 2

# ---------------------------------------------------------------------------- #
# TRAIN options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.EPOCHS = 20
_C.TRAIN.LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.VALIDATION_SPLIT = 0.2
_C.TRAIN.GRAD_CLIP_T = 1.

# ---------------------------------------------------------------------------- #
# Path options
# ---------------------------------------------------------------------------- #
_C.PATH = CN()
_C.PATH.ROOT = "/scratch/work/luongn1/transformer-cfr/"
_C.PATH.RESULT = "results/"
_C.PATH.MODEL_OUT_DIR = _C.PATH.ROOT + _C.PATH.RESULT + _C.MODEL.NAME 
_C.PATH.MIMIC_DATA_DIR = "/scratch/work/luongn1/transformer-cfr/data/mimic/processed/sequential_mimic.csv"

# ---------------------------------------------------------------------------- #
# Utils options
# ---------------------------------------------------------------------------- #
_C.UTILS = CN()

_C.UTILS.TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")