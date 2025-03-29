"""
Collection of utility functions used across the CAPO project.
Provides functionality for random seed management, hashing, object copying, and other common operations needed throughout the codebase.
"""

import copy
import hashlib
import os
import random
import string

import numpy as np
import torch


def generate_hash_from_string(text: str):
    """Generate a hash from a string."""
    hash_object = hashlib.sha256(text.encode())
    hash_string = hash_object.hexdigest()
    return hash_string


def generate_random_hash():
    """Generate a random hash."""
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=32))

    return generate_hash_from_string(random_string)


def seed_everything(seed: int = 42):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def copy_llm(model_obj, llm_attr_name="llm"):
    # Create a new instance of the same class
    new_obj = type(model_obj).__new__(type(model_obj))

    # Copy all attributes except the LLM
    for attr_name in model_obj.__dict__:
        if attr_name == llm_attr_name:
            # Direct reference assignment for LLM
            setattr(new_obj, attr_name, getattr(model_obj, attr_name))
        else:
            # Deep copy for other attributes
            setattr(new_obj, attr_name, copy.deepcopy(getattr(model_obj, attr_name)))

    return new_obj
