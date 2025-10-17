# project_pamana/03_code/utils.py

import os
import random
import numpy as np
import tensorflow as tf

# The fix is to use a direct import, which works because our scripts
# that call this module will add the '03_code' directory to the system path.
import config

def set_seeds():
    """
    Sets the random seeds for all relevant libraries to ensure reproducibility.
    It uses the SEED constant from the config.py file.
    """
    seed_value = config.SEED
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # The following line can be used for even stricter determinism in TensorFlow,
    # but it may have a performance cost.
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"All random seeds set to: {seed_value}")