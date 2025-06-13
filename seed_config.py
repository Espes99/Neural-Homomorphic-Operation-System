import numpy as np
import tensorflow as tf
import random

from learning_params import GLOBAL_SEED


def set_all_seeds(seed=GLOBAL_SEED):
    """
    Set seeds for all random number generators to ensure reproducibility

    Args:
        seed: The seed value to use. If None, uses GLOBAL_SEED
    """

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)