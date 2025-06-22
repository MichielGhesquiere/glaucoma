import random
import numpy as np
import torch
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Disabling benchmark and setting deterministic mode can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # logger.info("Note: cudnn.deterministic=True and benchmark=False set for reproducibility.")
    logger.info(f"Random seed set to {seed}")


class NpEncoder(json.JSONEncoder):
    """ Handles Numpy types for JSON serialization """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Handle NaN/inf gracefully for JSON
            if np.isnan(obj) or np.isinf(obj):
                return None # Represent as null in JSON
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)