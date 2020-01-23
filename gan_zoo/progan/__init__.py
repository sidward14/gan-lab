# -*- coding: UTF-8 -*-

"""Progressive Growing of GANs for Improved Quality, Stability, and Variation.
   (Karras et al. 2018)

Default configuration exactly emulates the most recent official implementation.

That being said, there are plenty of hyperparameter options/settings that you
can change in config.py to alter your ProGAN to your liking.
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