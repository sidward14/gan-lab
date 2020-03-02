# -*- coding: UTF-8 -*-

"""Base architecture class for non-progressive GANs.
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from abc import ABC, abstractmethod

import torch
from torch import nn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class GAN( nn.Module, ABC ):
  """Base GAN for all non-progressive architectures."""
  def __init__( self, res ):
    super( GAN, self ).__init__()
    self._res = res

  def most_parameters( self, recurse = True, excluded_params:list = [] ):
    """torch.nn.Module.parameters() generator method but with the option to exclude specified parameters."""
    for name, params in self.named_parameters( recurse = recurse ):
      if name not in excluded_params:
        yield params

  @property
  def res( self ):
    return self._res
  
  @res.setter
  def res( self, new_res ):
    message = f'GAN().res cannot be changed, as {self.__class__.__name__} only permits one resolution: {self._res}.'
    raise AttributeError( message )

  @abstractmethod
  def forward( self, x ):
    raise NotImplementedError( 'Can only call `forward` on valid subclasses.' )