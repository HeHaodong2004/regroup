FOLDER_NAME = 'advsearch'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SUMMARY_WINDOW = 2
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 1000

N_AGENTS = 3

EXPLORATION = True

CELL_SIZE = 0.4  # pixel to meter
NODE_RESOLUTION = 4.0 # node to meter
DOWNSAMPLE_SIZE = NODE_RESOLUTION // CELL_SIZE

# SENSOR_RANGE = 7.9812  # meter, for Gregorin maps
SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
SAFETY_RANGE = SENSOR_RANGE
MIN_UTILITY = 1
FRONTIER_CELL_SIZE = 4 * CELL_SIZE

LOCAL_MAP_SIZE = 40  # meter
EXTENDED_LOCAL_MAP_SIZE = 8 * SENSOR_RANGE * 1.05

MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 5000
BATCH_SIZE = 256
LR = 1e-5
GAMMA = 1

LOCAL_NODE_INPUT_DIM = 8
EMBEDDING_DIM = 128

LOCAL_K_SIZE = 25  # the number of neighboring nodes
LOCAL_NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 12
USE_WANDB = False
