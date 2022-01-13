import numpy as np
import torch
import os
import random
import torch.nn as nn

def fix_all_seeds(seed : int = 4496) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
def freeze_unfreeze(model : nn.Module, 
                    require_gradients : bool = True) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = require_gradients
    
    return model

def get_ground_truth_vector(classes : torch.Tensor, n_domains : int,
                            n_classes : int) -> torch.Tensor:
    """
    Get the ground truth vector for the phase where the feature extractor 
    tries that discriminator cannot distinguish the domain that the sample
    comes from.

    Args:
        classes (torch.Tensor): Class labels.
        n_domains (int): Number of domains.
        n_classes (int): Number of classes.

    Returns:
        torch.Tensor: Tensor containing the ground truth for each sample.
    """
    # Create the ground truth tensor
    total_size = n_domains * n_classes
    gt = torch.zeros(len(classes), total_size)

    # Value to be placed in the corresponding categories and domains positions
    # It is uniform so the discriminator cannot distinguish which domain the
    # sample comes from
    non_zero_value = 1 / n_domains
    
    for row in range(len(classes)):
        # The indices are the corresponding position for each class into each
        # domain        
        non_zero_indices = [i+classes[row] for i in range(0, total_size, n_classes)]
        gt[row, non_zero_indices] = non_zero_value
    
    return gt