from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'E2ECompressedGEBDModel'
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.DIM = 256
_C.MODEL.BACKBONE.SIDE_DATA_NAME = 'resnet18'
_C.MODEL.TEMPORAL_MODEL = 'transformer'  # Transformer or GRU
_C.MODEL.USE_GROUP_SIMILARITY = True  # Transformer or GRU

_C.MODEL.PRETRAINED = True
_C.MODEL.SYNC_BN = True
_C.MODEL.USE_GAN = False
_C.MODEL.USE_RESIDUAL = True
_C.MODEL.USE_MV_AS_DECONV_PARAMS = True
_C.MODEL.KERNEL_SIZE = 8

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ('train',)
_C.DATASETS.TEST = ('minval',)

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.FRAME_PER_SIDE = 5
_C.INPUT.DYNAMIC_DOWNSAMPLE = False
_C.INPUT.NO_DOWNSAMPLE = False
_C.INPUT.DOWNSAMPLE = 3
_C.INPUT.USE_SIDE_DATA = False
_C.INPUT.IMAGE_SIZE = 224
_C.INPUT.END_TO_END = False  # input whole video
_C.INPUT.USE_GOP = False  # using gop as unit
_C.INPUT.SEQUENCE_LENGTH = 50  # input whole video
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 30
_C.SOLVER.MILESTONES = [2, 3]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.AMPE = True  # automatic mixed precision training
_C.SOLVER.LR = 1e-2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.CLIP_GRAD = 0.0
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.OPTIMIZER = 'SGD'

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.THRESHOLD = 0.5
_C.TEST.PRED_FILE = ''  # precomputed predictions
_C.TEST.RELDIS_THRESHOLD = 0.05

_C.OUTPUT_DIR = 'output'
