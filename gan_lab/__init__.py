# -*- coding: UTF-8 -*-

"""
StyleGAN, ProGAN, and ResNet GANs with additional features such as supervised
learning capabilities, an easy-to-use interface for saving/loading the model state,
flexible learning rate scheduling (and re-scheduling) capabilities, and more.

Each GAN model's default settings exactly emulates its respective most recent
official implementation, but at the same time this package features a
simple interface (config.py) where the user can quickly tune an extensive
list of hyperparameter options/settings to their choosing.

This package is easy to use and the code is easy to read. It aims for an
intuitive API without sacrificing any complexity anywhere.
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from ._int import get_current_configuration

from . import utils
from . import resnetgan
from . import progan
from . import stylegan

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#