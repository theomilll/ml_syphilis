# utils.py
# Utility functions

import random
import numpy as np
import os

def seed_everything(seed=42):
    """Sets random seeds for reproducibility for Python, NumPy, and OS hash.
       Also attempts to seed PyTorch if available.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
            # The following lines can be uncommented if strict determinism is needed,
            # but they might impact performance.
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        print(f"PyTorch random seeds set to {seed}")
    except ImportError:
        print("torch not found, skipping PyTorch seed.")
    print(f"Global random seeds (Python, NumPy, OS hash) set to {seed}")

# As per project plan, individual files also contain seed lines:
# random.seed(42); np.random.seed(42); os.environ["PYTHONHASHSEED"]="42"
# This function (seed_everything) can be called from a main script or at the beginning
# of notebook execution for a more centralized approach if preferred later.

# Example of calling it if this utils.py is imported early:
# seed_everything(42)

print("utils.py loaded")
