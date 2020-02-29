# -*- coding: UTF-8 -*-

"""Custom layers.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from .initializer import Initializer

import torch
from torch import nn
import torch.nn.functional as F

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Lambda:
# -------

class Lambda( nn.Module ):
  """Converts any function into a PyTorch Module."""
  def __init__( self, func, **kwargs ):
    super( Lambda, self ).__init__()
    self.func = func
    if kwargs:
      self.kwargs = kwargs
    else:
      self.kwargs = {}

  def forward( self, x ):
    return self.func( x, **self.kwargs )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Blur (Low-pass Filtering):
# --------------------------

# TODO: Add More (such as Gaussian Blur)
def get_blur_op( blur_type, num_channels ):
  """Options for low-pass filter operations. Only 3x3 kernels supported currently."""
  if blur_type.casefold() == 'box':
    blur_filter = torch.FloatTensor( [ 1./9 ] ).expand( num_channels, 1, 3, 3 )
    stride = 3
  elif blur_type.casefold() == 'binomial':
    blur_filter = torch.FloatTensor( [ [ 1./16, 2./16, 1./16 ],
                                       [ 2./16, 4./16, 2./16 ],
                                       [ 1./16, 2./16, 1./16 ] ] ).expand( num_channels, 1, 3, 3 )
    stride = 1
  elif blur_type.casefold() == 'gaussian':
    raise NotImplementedError( 'Gaussian blur not yet implemented.' )

  # meant to work as a PyTorch Module
  blur_op = Lambda( lambda x: F.conv2d( x, blur_filter.to( x.device ),
                    stride = stride, padding = 1, groups = num_channels ) )

  return blur_op

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Pooling:
# --------

class NearestPool2d( nn.Module ):
  def __init__( self ):
    super( NearestPool2d, self ).__init__()
  def forward( self, x ):
    if x.dim() == 3:
      x = x.view( -1, *x.shape )
    return F.interpolate( x, scale_factor = .5, mode = 'nearest' )

class BilinearPool2d( nn.Module ):
  def __init__( self, align_corners ):
    super( BilinearPool2d, self ).__init__()
    self.align_corners = align_corners
  def forward( self, x ):
    if x.dim() == 3:
      x = x.view( -1, *x.shape )
    return F.interpolate( x, scale_factor = .5, mode = 'bilinear',
                          align_corners = self.align_corners )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Normalization:
# --------------

class PixelNorm2d( nn.Module ):
  def __init__( self ):
    super( PixelNorm2d, self ).__init__()

  def forward( self, x, eps = 1.e-8 ):
    return x * ( ( ( x**2 ).mean( dim = 1, keepdim = True ) + eps ).rsqrt() )

class NormalizeLayer( nn.Module ):
  """All normalization methods in one place."""
  # TODO: Implement Spectral Normalization
  def __init__( self, norm_type, ni = None, res = None ):
    super( NormalizeLayer, self ).__init__()

    norm_type = norm_type.lower( )

    if norm_type in ( 'pixelnorm', 'pixel norm', ):
      self.norm = PixelNorm2d( )
    elif norm_type in ( 'instancenorm', 'instance norm', ):
      self.norm = nn.InstanceNorm2d( ni, eps = 1.e-8 )
    elif norm_type in ( 'batchnorm', 'batch norm', ):
      assert isinstance( ni, int )
      self.norm = nn.BatchNorm2d( ni )
    elif norm_type in ( 'layernorm', 'layer norm', ):
      # self.norm = F.layer_norm
      assert isinstance( ni, int ); assert isinstance( res, int )
      self.norm = nn.LayerNorm( [ ni, res, res ] )
    else:
      raise Exception( f'`norm_type` == "{norm_type}" not supported.' )

  def forward( self, x ):
      return self.norm( x )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Minibatch Standard Deviation:
# -----------------------------

def concat_mbstd_layer( x, group_size = 4 ):
  """Minibatch Standard Deviation layer."""
  _sz = x.size()

  # `group_size` must be less than or equal to and divisible into minibatch size;
  # otherwise use the entire minibatch
  group_size = min( _sz[0], group_size )
  if _sz[0] % group_size != 0:
    group_size = _sz[0]
  G = _sz[0] // group_size

  if group_size > 1:
    mbstd_map = x.view( G, group_size, _sz[1], _sz[2], _sz[3] )
    mbstd_map = torch.var( mbstd_map, dim = 1 )
    mbstd_map = torch.sqrt( mbstd_map + 1.e-8 )
    mbstd_map = mbstd_map.view( G, -1 )
    mbstd_map = torch.mean( mbstd_map, dim = 1 ).view( G, 1 )
    mbstd_map = mbstd_map.expand( G, _sz[2]*_sz[3] ).view( ( G, 1, 1, _sz[2], _sz[3] ) )
    mbstd_map = mbstd_map.expand( G, group_size, -1, -1, -1 )
    mbstd_map = mbstd_map.contiguous().view( ( -1, 1, _sz[2], _sz[3] ) )
  else:
    mbstd_map = torch.zeros( _sz[0], 1, _sz[2], _sz[3], device = x.device )

  return torch.cat( ( x, mbstd_map, ), dim = 1 )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Bread-and-butter Linear Operations, but with features such as
# equalized LR, custom initialization, etc:
# ------------------------------------------------------------

class Conv2dEx( nn.Module ):
  def __init__( self, ni, nf, ks, stride = 1, padding = 0, groups = 1, init = 'he', init_type = 'default',
                gain_sq_base = 2., equalized_lr = False, lrmul = 1., include_bias = True ):
    super( Conv2dEx, self ).__init__()

    self.ni = ni; self.nf = nf

    init = init.casefold()
    init_type = init_type.casefold()
    self.equalized_lr = equalized_lr
    self.use_lrmul = True if lrmul != 1. else False
    self.lrmul = lrmul

    self.initializer = None
    if init_type != 'standard normal':
      self.initializer = Initializer( init = init, init_type = init_type,
                                      gain_sq_base = gain_sq_base,
                                      equalized_lr = equalized_lr )

    self.conv2d = nn.Conv2d(
      ni, nf, kernel_size = ks, stride = stride,
      padding = padding, groups = groups, bias = include_bias
    )

    # Weights:
    self.wscale = None
    if init_type == ( 'default', 'resnet', ) and init is not None:
      bound = self.initializer.get_init_bound_layer(
        tensor = self.conv2d.weight,
        distribution_type = 'Uniform', stride = stride
      )
      if equalized_lr:
        self.wscale = bound
        self.conv2d.weight.data.uniform_( -1. / lrmul, 1. / lrmul )
      else:
        self.conv2d.weight.data.uniform_( -bound / lrmul, bound / lrmul )
    elif init_type in ( 'progan', 'stylegan', ) and init is not None:
      bound = self.initializer.get_init_bound_layer(
        tensor = self.conv2d.weight,
        distribution_type = 'Normal', stride = stride
      )
      if equalized_lr:
        self.wscale = bound
        self.conv2d.weight.data.normal_( 0., 1. / lrmul )
      else:
        self.conv2d.weight.data.normal_( 0., bound / lrmul )
    elif init_type == 'standard normal' and init is None and not equalized_lr and not self.use_lrmul:
      self.conv2d.weight.data.normal_( 0., 1. )

    # Biases:
    self.bias = None
    if include_bias:
      # init biases as 0, according to official implementations.
      self.conv2d.bias.data.fill_( 0 )

  def forward( self, x ):
    if self.equalized_lr:
      x = self.conv2d( x.mul( self.wscale ) )
    else:
      x = self.conv2d( x )

    if self.use_lrmul:
      x *= self.lrmul

    return x

class Conv2dBias( nn.Module ):
  def __init__( self, nf, lrmul = 1., device = 'cuda' if torch.cuda.is_available() else 'cpu' ):
    super( Conv2dBias, self ).__init__()

    self.use_lrmul = True if lrmul != 1. else False
    self.lrmul = lrmul

    self.bias = nn.Parameter( torch.FloatTensor( 1, nf, 1, 1 ).to( device ).fill_( 0 ) )

  def forward( self, x ):
    if self.use_lrmul:
      return x + self.bias.mul( self.lrmul )
    else:
      return x + self.bias

# ............................................................................ #

class LinearEx( nn.Module ):
  def __init__( self, nin_feat, nout_feat, init = 'xavier', init_type = 'default',
                gain_sq_base = 2., equalized_lr = False, lrmul = 1., include_bias = True ):
    super( LinearEx, self ).__init__()

    self.nin_feat = nin_feat; self.nout_feat = nout_feat

    init = init.casefold()
    init_type = init_type.casefold()
    self.equalized_lr = equalized_lr
    self.use_lrmul = True if lrmul != 1. else False
    self.lrmul = lrmul

    self.initializer = None
    if init_type != 'standard normal':
      self.initializer = Initializer( init = init, init_type = init_type,
                                      gain_sq_base = gain_sq_base,
                                      equalized_lr = equalized_lr )

    self.linear = nn.Linear( nin_feat, nout_feat, bias = include_bias )

    # Weights:
    self.wscale = None
    if init_type in ( 'default', 'resnet', ) and init is not None:
      bound = self.initializer.get_init_bound_layer(
        tensor = self.linear.weight,
        distribution_type = 'Uniform'
      )
      if equalized_lr:
        self.wscale = bound
        self.linear.weight.data.uniform_( -1. / lrmul, 1. / lrmul )
      else:
        self.linear.weight.data.uniform_( -bound / lrmul, bound / lrmul )
    elif init_type in ( 'progan', 'stylegan', ) and init is not None:
      bound = self.initializer.get_init_bound_layer(
        tensor = self.linear.weight,
        distribution_type = 'Normal'
      )
      if equalized_lr:
        self.wscale = bound
        self.linear.weight.data.normal_( 0., 1. / lrmul )
      else:
        self.linear.weight.data.normal_( 0., bound / lrmul )
    elif init_type == 'standard normal' and init is None and not equalized_lr and not self.use_lrmul:
      self.linear.weight.data.normal_( 0., 1. )

    # Biases:
    self.bias = None
    if include_bias:
      # init biases as 0, according to official implementations.
      self.linear.bias.data.fill_( 0 )

  def forward( self, x ):
    if self.equalized_lr:
      x = self.linear( x.mul( self.wscale ) )
    else:
      x = self.linear( x )

    if self.use_lrmul:
      x *= self.lrmul

    return x

class LinearBias( nn.Module ):
  def __init__( self, nout_feat, lrmul = 1., device = 'cuda' if torch.cuda.is_available() else 'cpu' ):
    super( LinearBias, self ).__init__()

    self.use_lrmul = True if lrmul != 1. else False
    self.lrmul = lrmul

    self.bias = nn.Parameter( torch.FloatTensor( 1, nout_feat ).to( device ).fill_( 0 ) )

  def forward( self, x ):
    if self.use_lrmul:
      return x + self.bias.mul( self.lrmul )
    else:
      return x + self.bias