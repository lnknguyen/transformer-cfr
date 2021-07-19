from yacs.config import CfgNode as CN

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
_C.MODEL.TYPE = ""

_C.MODEL.EMBEDDING_DIM = 300
_C.MODEL.DROPOUT_P = 0.0
_C.MODEL.ENCODER_NUM_LAYER = 2

_C.MODEL.DIAG_INPUT_VOCAB_SIZE = 1000
_C.MODEL.PROC_INPUT_VOCAB_SIZE = 1000
_C.MODEL.VOCAB_SIZE = 258

_C.MODEL.N_OUT_CLASSES = 2

_C.MODEL.RNN_HIDDEN_SIZE = 64
_C.MODEL.CAT_RNN_HIDDEN_SIZE = 10
_C.MODEL.FC_HIDDEN_SIZE = 16

# Whether to not use icd and proc codes for trainig
_C.MODEL.SWITCHOFF_CODES = False
_C.MODEL.IS_BIDIRECTIONAL = False

_C.MODEL.MAX_SEQ_LENGTH = 200

# Transformer Params
_C.MODEL.NUM_HEADS = 6

_C.MODEL.TRANS_DEPTH = 2

