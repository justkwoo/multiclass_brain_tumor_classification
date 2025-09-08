import torch

# define device
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
# define data dir
TRAIN_DATA_DIR = f'../data/training'
TEST_DATA_DIR = f'../data/testing'
NEW_DATA_DIR = f'../new_data'

# transformation hyperparameter
IMG_RESIZE_DIM = (224, 224) 

# dataloader parameters    
BATCH_SIZE = 32
NUM_WORKERS = 8 # change it to 4 or 6 if draining computational power too much
PREFETCH_FACTOR = 4

# CNN hyperparameters
FC_LAYER_SIZE = 1024
DROPOUT_RATE = 0.5

# training hyperparameters
NUM_CLASSES = 4
LEARNING_RATE = 0.001 
EPOCHS = 250 
PATIENCE = 4

# define path to save model
DIR_TO_SAVE_MODEL = '../saved_models'

# define path to save train result
DIR_TO_TRAIN_RESULT = '../train_result/train_valid_result'
DIR_TO_BEST_EPOCH = '../train_result/best_epoch'