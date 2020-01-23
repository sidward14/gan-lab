# -*- coding: UTF-8 -*-

"""Performs initialization for almost all parameters in the neural networks of this package.

Handles multiple kinds of parameter initializations from multiple kinds of
models (ResNet GAN, ProGAN, StyleGAN, etc.).
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

import numpy as np
import torch

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class Initializer( object ):
  """Initialization class that handles multiple kinds of parameter initializations."""
  def __init__( self, init, init_type = 'default', gain_sq_base = 2., equalized_lr = False ):
    super( Initializer, self ).__init__()

    self.init = init.casefold()
    self.init_type = init_type.casefold()
    self.gain_sq_base = gain_sq_base
    self.equalized_lr = equalized_lr

  @torch.no_grad()
  def get_init_bound_layer( self, tensor, distribution_type, stride = 1 ):
    distribution_type = distribution_type.casefold()
    if distribution_type not in ( 'uniform', 'normal', ):
      raise ValueError( 'Only uniform and normal distributions are supported.' )

    if self.init_type in ( 'default', 'resnet', ):
      fan_in, fan_out = self._calculate_fan_in_fan_out( tensor = tensor, stride = stride )
      std = self._calculate_init_weight_std( fan_in = fan_in, fan_out = fan_out )
    elif self.init_type in ( 'progan', 'stylegan', ):
      fan_in, _ = self._calculate_fan_in_fan_out( tensor = tensor, stride = stride )
      std = self._calculate_init_weight_std( fan_in = fan_in )

    return np.sqrt( 3 )*std if distribution_type == 'uniform' else std

  @torch.no_grad()
  def _calculate_fan_in_fan_out( self, tensor, stride = 1 ):
    dimensions = tensor.dim()
    if dimensions < 2:
      raise ValueError(
        'Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions.'
      )

    fan_out = None
    if dimensions == 2:  # Linear
      fan_in = tensor.size(1)
      if self.init_type not in ( 'progan', 'stylegan', ):
        fan_out = tensor.size(0)
    else:
      receptive_field_size = 1
      if dimensions > 2:
        receptive_field_size = tensor[0][0].numel()
      fan_in = tensor.size(1) * receptive_field_size
      if self.init_type not in ( 'progan', 'stylegan', ):
        fan_out = tensor.size(0) * receptive_field_size / stride**2

    return fan_in, fan_out

  @torch.no_grad()
  def _calculate_init_weight_std( self, fan_in = None, fan_out = None ):
    gain_sq = self.gain_sq_base / 2.
    if fan_out is not None and fan_in is not None:
      fan = fan_in + fan_out
      gain_sq *= 2
    elif fan_in is not None:
      fan = fan_in
    elif fan_out is not None:
      fan = fan_out

    if self.init == 'he': gain_sq = 2. * gain_sq
    elif self.init == 'xavier': gain_sq = 1. * gain_sq

    std = np.sqrt( gain_sq / fan )

    return std