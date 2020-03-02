# -*- coding: UTF-8 -*-

"""ResBlocks used to construct ResNet GANs.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from utils.custom_layers import Lambda, get_blur_op, NormalizeLayer, Conv2dEx

import torch
from torch import nn

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class ResBlock2d( nn.Module ):
  def __init__( self, ni, nf, ks, norm_type, upsampler = None, pooler = None,
                init = 'He', nl = nn.ReLU(), res = None, flip_sampling = False,
                equalized_lr = False, blur_type = None ):
    super( ResBlock2d, self ).__init__()

    assert not ( upsampler is not None and pooler is not None )

    padding = ( ks - 1 ) // 2  # 'SAME' padding for stride 1 conv

    if not flip_sampling:
      self.nif = nf if ( upsampler is not None and pooler is None ) else ni
    else:
      self.nif = ni if ( upsampler is None and pooler is not None ) else nf
    self.convs = (
      Conv2dEx( ni, self.nif, ks = ks, stride = 1, padding = padding,
                init = init, equalized_lr = equalized_lr ),
      Conv2dEx( self.nif, nf, ks = ks, stride = 1, padding = padding,
                init = init, equalized_lr = equalized_lr ),
      Conv2dEx( ni, nf, ks = 1, stride = 1, padding = 0,
                init = 'Xavier', equalized_lr = equalized_lr ),  # this is same as a FC layer
    )

    blur_op = get_blur_op( blur_type = blur_type, num_channels = self.convs[0].nf ) if blur_type is not None else None

    _norm_nls = (
      [ NormalizeLayer( norm_type, ni = ni, res = res ), nl ],
      [ NormalizeLayer( norm_type, ni = self.convs[0].nf, res = res ), nl ],
    )

    if upsampler is not None:
      _mostly_linear_op_1 = [ upsampler, self.convs[0], blur_op ] if blur_type is not None else [ upsampler, self.convs[0] ]
      _mostly_linear_op_2 = [ upsampler, self.convs[2], blur_op ] if blur_type is not None else [ upsampler, self.convs[2] ]
      _ops = ( _mostly_linear_op_1, [ self.convs[1] ], _mostly_linear_op_2, )
    elif pooler is not None:
      _mostly_linear_op_1 = [ blur_op, self.convs[1], pooler ] if blur_type is not None else [ self.convs[1], pooler ]
      _mostly_linear_op_2 = [ blur_op, pooler, self.convs[2] ] if blur_type is not None else [ pooler, self.convs[2] ]
      _ops = ( [ self.convs[0] ], _mostly_linear_op_1, _mostly_linear_op_2, )
    else:
      _ops = ( [ self.convs[0] ], [ self.convs[1] ], [ self.convs[2] ], )

    self.conv_layer_1 = nn.Sequential( *( _norm_nls[0] + _ops[0] ) )
    self.conv_layer_2 = nn.Sequential( *( _norm_nls[1] + _ops[1] ) )

    if ( upsampler is not None or pooler is not None ) or ni != nf:
      self.skip_connection = nn.Sequential( *( _ops[2] ) )
    else:
      self.skip_connection = Lambda( lambda x: x )

  def forward( self, x ):
    return self.skip_connection( x ) + self.conv_layer_2( self.conv_layer_1( x ) )


class ResBlock2d32Pix( ResBlock2d ):
  def __init__( self, ni, nf, ks, norm_type, upsampler = None, pooler = None,
                init = 'He', nl = nn.ReLU(), res = None, flip_sampling = True,
                equalized_lr = False, blur_type = None ):
    super( ResBlock2d32Pix, self ).__init__( ni, nf, ks, norm_type, upsampler, pooler, init,
                                             nl, res, flip_sampling, equalized_lr, blur_type )

    if upsampler is None and pooler is not None:
      self.skip_connection = nn.Sequential( self.convs[2], pooler )

  def forward( self, x ):
    return self.skip_connection( x ) + self.conv_layer_2( self.conv_layer_1( x ) )


class FastResBlock2dDownsample( nn.Module ):
  """Downsampling ResBlock without normalization and activation after first conv2d."""
  def __init__( self, ni, nf, ks, pooler = nn.AvgPool2d( kernel_size = 2, stride = 2 ),
                init = 'He', nl = nn.ReLU(), equalized_lr = False, blur_type = None ):
    super( FastResBlock2dDownsample, self ).__init__()

    padding = ( ks - 1 ) // 2

    self.conv_layer_1 = nn.Sequential(
      Conv2dEx( ni, nf, ks = ks, stride = 1, padding = padding,
                init = 'he', equalized_lr = equalized_lr ),
      nl
    )

    self.conv_layer_2 = nn.Sequential( )
    self.skip_connection = nn.Sequential( )

    _seq_n = 0
    if blur_type is not None:
      blur_op = get_blur_op( blur_type = blur_type, num_channels = nf )

      self.conv_layer_2.add_module( str( _seq_n ), blur_op )
      self.skip_connection.add_module( str( _seq_n ), blur_op )
      _seq_n += 1

    self.conv_layer_2.add_module(
      str( _seq_n ), Conv2dEx( nf, nf, ks = ks, stride = 1, padding = padding,
                               init = 'he', equalized_lr = equalized_lr )
    )
    self.skip_connection.add_module( str( _seq_n ), pooler )
    _seq_n += 1

    self.conv_layer_2.add_module( str( _seq_n ), pooler )
    self.skip_connection.add_module(
      str( _seq_n ), Conv2dEx( ni, nf, ks = 1, stride = 1, padding = 0,
                               init = 'xavier', equalized_lr = equalized_lr )
    )

  def forward( self, x ):
    return self.skip_connection( x ) + self.conv_layer_2( self.conv_layer_1( x ) )