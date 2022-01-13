import torch

class Config:
    IMGS_PATH = '/home/data'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    N_DOMAINS = 11
    N_CLASSES = 2
    
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10000
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.000000001
    MOMENTUM = 0.9