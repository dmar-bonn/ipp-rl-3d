BATCH_SIZE = 64
INPUT_DIM = 6 # 7 if including binary
EMBEDDING_DIM = 128
N_HEADS = 4
ACCUMULATION_STEPS = 1024

K_SIZE = 20
FOLDER_NAME = 'k20_novec_final'

PLANT_RANGE = (10, 15)
BUDGET_RANGE = (7.0, 9.0)
SAMPLE_LENGTH = 0.2
GEN_RANGE = (0.1, 0.3) # Distance range for generating new samples
SAMPLING_SIZE = K_SIZE

PRED_GRID = 50

ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4
BETA = 1 # the certainty below mean_val a cell must possess before being considered interesting

LENGTH_SCALE = 0.5

DEPTH = 12
FACING_ACTIONS = ['F', 'B', 'L', 'R'] # N, S, W, E

RUN_TYPE = 'prev_samples'
EVAL_TYPE = 'cov_tr'
TRIAL_NUM = '0'

USE_GPU = False
USE_GPU_GLOBAL = False # True if GPU available
CUDA_DEVICE = [0]

NUM_META_AGENT = 12
LR = 1e-4
GAMMA = 1
EPSILON = 2e-1
DECAY_STEP = 32
SUMMARY_WINDOW = 1

TREE_BINARY = False
DIST_SAMPLE = True
EXPLORATION_ONLY = False

model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
logs_path = f'logs/{FOLDER_NAME}'
LOAD_MODEL = True
SAVE_IMG_GAP = 500
FRUIT_EMPHASIS = True
