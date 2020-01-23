# -*- coding: UTF-8 -*-

"""A Style-Based Generator Architecture for Generative Adversarial Networks.
   (Karras et al. 2019)

Default configuration exactly emulates the most recent official implementation.

That being said, there are plenty of hyperparameter options/settings that you
can change in config.py to alter your StyleGAN to your liking.
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import os; import sys
sys.path.append(
  os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..' ) )
)

from . import base
from . import architectures
from . import learner

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#