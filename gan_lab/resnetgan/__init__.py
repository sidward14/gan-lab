# -*- coding: UTF-8 -*-

"""ResNet GANs and non-progressive GANs in general.
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import os; import sys
sys.path.append(
  os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..' ) )
)

from . import resblocks
from . import base
from . import architectures
from . import learner

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#