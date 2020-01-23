# -*- coding: UTF-8 -*-

"""Base architecture class for ProGANs (Progressively Growing GANs).
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

FMAP_BASE = 8192
FMAP_MAX = 512

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class ProGAN( nn.Module, ABC ):
  """This base class updates and keeps track of metadata that define the current
     state of the ProGAN model.
  """

  # these class attributes should be updated at the ProGANLearner class level

  _alpha = 1  # fading-in parameter
  _alpha_tol = 1.e-8

  _final_res = None
  _prev_res = None
  _curr_res = 4
  _scale_stage = int( np.log2( _curr_res ) ) - 1
  _fmap_prev = None
  _fmap = min( int( FMAP_BASE / ( 2**_scale_stage ) ), FMAP_MAX )
  _scale_inc_metadata_updated = False
  _fade_in_phase = False

  @classmethod
  def reset_state( cls ):
    """Call this to reset ProGAN state in case one wants to start over."""
    cls._alpha = 1
    cls._alpha_tol = 1.e-8

    cls._prev_res = None
    cls._curr_res = 4
    if cls._final_res is not None: assert cls._curr_res <= cls._final_res
    cls._scale_stage = int( np.log2( cls._curr_res ) ) - 1
    cls._fmap_prev = None
    cls._fmap = min( int( FMAP_BASE / ( 2**cls._scale_stage ) ), FMAP_MAX )
    cls._scale_inc_metadata_updated = False
    cls._fade_in_phase = False

  def __init__( self, final_res ):
    self.cls_base = self.__class__.__base__
    super( self.cls_base, self ).__init__()

    self.final_res = final_res
    assert self.curr_res <= self.final_res

  def increase_scale( self ):
    """Use this to increase scale during training or for initial resolution."""
    self.prev_res = self.curr_res
    self.curr_res = int( 2**( int( np.log2( self.curr_res ) ) + 1 ) )

    self.scale_stage = int( np.log2( self.curr_res ) ) - 1  # done this way for readability and congruency

    self.fmap_prev = self.fmap
    self.fmap = self.get_fmap( scale_stage = self.scale_stage )

    self.scale_inc_metadata_updated = True

    self.fade_in_phase = True

  def get_fmap( self, scale_stage ):
    return min( int( FMAP_BASE / ( 2**scale_stage ) ), FMAP_MAX )

  def most_parameters( self, recurse = True, excluded_params:list = [] ):
    """`torch.nn.Module.parameters()` generator method but with the option to exclude specified parameters."""
    for name, params in self.named_parameters( recurse = recurse ):
      if name not in excluded_params:
        yield params

  @property
  def fade_in_phase( self ):
    return self.cls_base._fade_in_phase

  @fade_in_phase.setter
  def fade_in_phase( self, new_fade_in_phase ):
    self.cls_base._fade_in_phase = new_fade_in_phase

  @property
  def scale_inc_metadata_updated( self ):
    return self.cls_base._scale_inc_metadata_updated

  @scale_inc_metadata_updated.setter
  def scale_inc_metadata_updated( self, new_scale_inc_metadata_updated ):
    self.cls_base._scale_inc_metadata_updated = new_scale_inc_metadata_updated

  @property
  def fmap( self ):
    return self.cls_base._fmap

  @fmap.setter
  def fmap( self, new_fmap ):
    self.cls_base._fmap = new_fmap

  @property
  def fmap_prev( self ):
    return self.cls_base._fmap_prev

  @fmap_prev.setter
  def fmap_prev( self, new_fmap_prev ):
    self.cls_base._fmap_prev = new_fmap_prev

  @property
  def scale_stage( self ):
    return self.cls_base._scale_stage

  @scale_stage.setter
  def scale_stage( self, new_scale_stage ):
    self.cls_base._scale_stage = new_scale_stage

  @property
  def curr_res( self ):
    return self.cls_base._curr_res

  @curr_res.setter
  def curr_res( self, new_curr_res ):
    self.cls_base._curr_res = new_curr_res

  @property
  def final_res( self ):
    return self.cls_base._final_res

  @final_res.setter
  def final_res( self, new_final_res ):
    self.cls_base._final_res = new_final_res

  @property
  def prev_res( self ):
    return self.cls_base._prev_res

  @prev_res.setter
  def prev_res( self, new_prev_res ):
    self.cls_base._prev_res = new_prev_res

  @property
  def alpha_tol( self ):
    return self.cls_base._alpha_tol

  @alpha_tol.setter
  def alpha_tol( self, new_alpha_tol ):
    self.cls_base._alpha_tol = new_alpha_tol

  @property
  def alpha( self ):
    return self.cls_base._alpha

  @alpha.setter
  def alpha( self, new_alpha ):
    if not ( 0. <= new_alpha < 1. + self.alpha_tol ):
      raise ValueError( 'Input alpha parameter must be in the range [0,1].' )

    if 1. - self.alpha_tol < new_alpha < 1. + self.alpha_tol:
      self.fade_in_phase = False
      self.cls_base._alpha = 1
    else:
      self.cls_base._alpha = new_alpha

  @abstractmethod
  def forward( self, x ):
    raise NotImplementedError( 'Can only call `forward` on valid subclasses.' )