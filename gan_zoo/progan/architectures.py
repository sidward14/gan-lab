# -*- coding: UTF-8 -*-

"""ProGAN architectures (generator and discriminator).
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from .base import ProGAN
from _int import FMAP_SAMPLES, RES_INIT
# from stylegan.base import StyleGAN
from utils.custom_layers import Lambda, get_blur_op, NormalizeLayer, \
                                concat_mbstd_layer, Conv2dEx, LinearEx, \
                                Conv2dBias

import copy

import torch
from torch import nn
import torch.nn.functional as F

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

FMAP_G_INIT_FCTR = 1
FMAP_D_END_FCTR = 1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Generator:
# ----------

# TODO: Implement more efficient "recursive structure" by Tero Karras for training (see their GitHub implementation)
class ProGenerator( ProGAN ):
  """Progressively Growing GAN (Karras et al. 2018) Generator.

  Emulates most recent official implementation.
  """
  def __init__( self,
                final_res,
                len_latent = 512,
                upsampler = nn.Upsample( scale_factor = 2, mode = 'nearest' ),
                blur_type = None,
                nl = nn.LeakyReLU( negative_slope = .2 ),
                num_classes = 0,
                equalized_lr = True,
                normalize_z = True,
                use_pixelnorm = True ):

    super( self.__class__, self ).__init__( final_res )

    self.gen_blocks = nn.ModuleList( )

    self.upsampler = upsampler
    self.upsampler_skip_connection = \
      lambda xb: F.interpolate( xb, scale_factor = 2, mode = 'nearest' )  # keep fading-in layers simple

    self.gen_blur_type = blur_type

    self.nl = nl

    self.len_latent = len_latent
    self.num_classes = num_classes

    self.equalized_lr = equalized_lr

    norms = []
    self.use_pixelnorm = use_pixelnorm
    if use_pixelnorm:
      norms.append( NormalizeLayer( 'PixelNorm' ) )

    if normalize_z:
      self.preprocess_z = nn.Sequential(
        Lambda( lambda x: x.view( -1, len_latent + num_classes ) ),
        NormalizeLayer( 'PixelNorm' )
      )
    else:
      self.preprocess_z = Lambda( lambda x: x.view( -1, len_latent + num_classes ) )

    _fmap_init = len_latent * FMAP_G_INIT_FCTR
    self.gen_blocks.append(
      nn.Sequential(
        LinearEx( nin_feat = len_latent + num_classes,
                  nout_feat = _fmap_init * RES_INIT**2,
                  init = 'He', init_type = 'ProGAN', gain_sq_base = 2./16,
                  equalized_lr = equalized_lr ),  # this can be done with a tranpose conv layer as well (efficiency)
        Lambda( lambda x: x.view( -1, _fmap_init, RES_INIT, RES_INIT ) ),
        nl,
        *norms,
        Conv2dEx( ni = _fmap_init, nf = self.fmap, ks = 3,
                  stride = 1, padding = 1, init = 'He', init_type = 'ProGAN',
                  gain_sq_base = 2., equalized_lr = equalized_lr ),
        nl,
        *norms
      )
    )

    self.prev_torgb = None
    self._update_torgb( ni = self.fmap )

  def increase_scale( self ):
    """Use this to increase scale during training or for initial resolution."""
    # update metadata
    if not self.scale_inc_metadata_updated:
      super( self.__class__, self ).increase_scale()
    else:
      self.scale_inc_metadata_updated = False

    blur_op = get_blur_op( blur_type = self.gen_blur_type, num_channels = self.fmap ) if \
              self.gen_blur_type is not None else None

    self.gen_blocks.append(
      nn.Sequential(
        self.get_conv_layer( ni = self.fmap_prev, upsample = True, blur_op = blur_op ),
        self.get_conv_layer( ni = self.fmap )
      )
    )

    self.prev_torgb = copy.deepcopy( self.torgb )
    self._update_torgb( ni = self.fmap )

  def get_conv_layer( self, ni, upsample = False, blur_op = None, append_nl = True ):
    upsampler = []
    if upsample:
      upsampler.append( self.upsampler )

    if blur_op is not None:
      conv = Conv2dEx( ni = ni, nf = self.fmap, ks = 3, stride = 1, padding = 1,
                       init = 'He', init_type = 'ProGAN', gain_sq_base = 2.,
                       equalized_lr = self.equalized_lr, include_bias = False )

      assert isinstance( blur_op, nn.Module )
      blur = [ blur_op ]

      bias = [ Conv2dBias( nf = self.fmap ) ]
    else:
      conv = Conv2dEx( ni = ni, nf = self.fmap, ks = 3, stride = 1, padding = 1,
                       init = 'He', init_type = 'ProGAN', gain_sq_base = 2.,
                       equalized_lr = self.equalized_lr, include_bias = True )
      blur = []
      bias = []

    nl = []
    if append_nl:
      nl.append( self.nl )

    norm = []
    if self.use_pixelnorm:
      norm.append( NormalizeLayer( 'PixelNorm'  ) )

    return nn.Sequential( *upsampler, conv, *( blur + bias + nl + norm ) )

  def _update_torgb( self, ni ):
    self.torgb = Conv2dEx( ni = ni, nf = FMAP_SAMPLES, ks = 1, stride = 1,
                           padding = 0, init = 'He', init_type = 'ProGAN',
                           gain_sq_base = 1., equalized_lr = self.equalized_lr )

  # def get_fmap( self, scale_stage ):
  #   return min( int( FMAP_BASE / ( 2**scale_stage ) ), FMAP_MAX )

  # NOTE: Perhaps remove for loop and ideally just make it all one function (i.e. `return self.func( x )`)
  def forward( self, x ):
    x = self.preprocess_z( x )
    for gen_block in self.gen_blocks[ :-1 ]:
      x = gen_block( x )
    if self.fade_in_phase:
      return self.upsampler_skip_connection( self.prev_torgb( x ) ) * ( 1. - self.alpha ) + \
             self.torgb( self.gen_blocks[ -1 ]( x ) ) * ( self.alpha )
    else:
      return self.torgb( self.gen_blocks[ -1 ]( x ) )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Discriminator:
# --------------

# TODO: Implement more efficient "recursive structure" by Tero Karras for training (see their GitHub implementation)
# TODO: Implement auxiliary classifier for ProGAN (AC-ProGAN)
class ProDiscriminator( ProGAN ):
  """Progressively Growing GAN (Karras et al. 2018) Discriminator/Critic.

  Emulates most recent official implementation.
  """
  def __init__( self,
                final_res,
                pooler = nn.AvgPool2d( kernel_size = 2, stride = 2 ),
                blur_type = None,
                nl = nn.LeakyReLU( negative_slope = .2 ),
                num_classes = 0,
                equalized_lr = True,
                mbstd_group_size = 4 ):

    super( self.__class__, self ).__init__( final_res )
    self.init_type = self.cls_base.__name__
    if self.init_type not in ( 'ProGAN', 'StyleGAN', ):
      raise RuntimeError( 'This class can only inherit from either `ProGAN` or `StyleGAN` base classes currently.' )

    self.disc_blocks = nn.ModuleList( )

    self.num_classes = num_classes

    self.preprocess_x = Lambda(
      lambda x: x.view( -1, FMAP_SAMPLES + num_classes, self.curr_res, self.curr_res )
    )

    self.pooler = pooler
    self.pooler_skip_connection = \
      lambda xb: F.avg_pool2d( xb, kernel_size = 2, stride = 2 )  # keep fading-in layers simple

    self.disc_blur_type = blur_type

    self.nl = nl

    self.equalized_lr = equalized_lr

    self.mbstd_group_size = mbstd_group_size
    mbstd_layer = self.get_mbstd_layer( )

    self.prev_fromrgb = None
    self._update_fromrgb( nf = self.fmap )

    _fmap_end = self.fmap * FMAP_D_END_FCTR
    self.disc_blocks.insert( 0,
      nn.Sequential(
        *mbstd_layer,
        Conv2dEx( ni = self.fmap + ( 1 if mbstd_layer else 0 ), nf = self.fmap, ks = 3, stride = 1,
                  padding = 1, init = 'He', init_type = self.init_type, gain_sq_base = 2.,
                  equalized_lr = equalized_lr ),  # this can be done with a linear layer as well (efficiency)
        nl,
        Conv2dEx( ni = self.fmap, nf = _fmap_end, ks = 4,
                  stride = 1, padding = 0, init = 'He', init_type = self.init_type,
                  gain_sq_base = 2., equalized_lr = equalized_lr ),
        nl,
        Lambda( lambda x: x.view( -1, _fmap_end ) ),
        LinearEx( nin_feat = _fmap_end, nout_feat = 1, init = 'He',
                  init_type = self.init_type, gain_sq_base = 1., equalized_lr = equalized_lr )
      )
    )

  def increase_scale( self ):
    """Use this to increase scale during training or for initial resolution."""
    # update metadata
    if not self.scale_inc_metadata_updated:
      super( self.__class__, self ).increase_scale()
    else:
      self.scale_inc_metadata_updated = False

    self.preprocess_x = Lambda(
      lambda x: x.view( -1, FMAP_SAMPLES + self.num_classes, self.curr_res, self.curr_res )
    )

    self.prev_fromrgb = copy.deepcopy( self.fromrgb )
    self._update_fromrgb( nf = self.fmap )

    blur_op = get_blur_op( blur_type = self.disc_blur_type, num_channels = self.fmap ) if \
              self.disc_blur_type is not None else None

    self.disc_blocks.insert( 0,
      nn.Sequential(
        self.get_conv_layer( nf = self.fmap ),
        self.get_conv_layer( nf = self.fmap_prev, downsample = True, blur_op = blur_op )
      )
    )

  def get_conv_layer( self, nf, downsample = False, blur_op = None, append_nl = True ):
    blur = []
    if blur_op is not None:
      assert isinstance( blur_op, nn.Module )
      blur.append( blur_op )

    if downsample:
      conv = Conv2dEx( ni = self.fmap, nf = nf, ks = 3, stride = 1, padding = 1,
                       init = 'He', init_type = self.init_type, gain_sq_base = 2.,
                       equalized_lr = self.equalized_lr, include_bias = False )
      pooler = [ self.pooler ]
      bias = [ Conv2dBias( nf = nf ) ]
    else:
      conv = Conv2dEx( ni = self.fmap, nf = nf, ks = 3, stride = 1, padding = 1,
                       init = 'He', init_type = self.init_type, gain_sq_base = 2.,
                       equalized_lr = self.equalized_lr, include_bias = True )
      pooler = []
      bias = []

    nl = []
    if append_nl:
      nl.append( self.nl )

    return nn.Sequential( *blur, conv, *( pooler + bias + nl ) )

  def _update_fromrgb( self, nf ):
    self.fromrgb = nn.Sequential(
      Conv2dEx( ni = FMAP_SAMPLES + self.num_classes, nf = nf, ks = 1, stride = 1,
                padding = 0, init = 'He', init_type = self.init_type,
                gain_sq_base = 2., equalized_lr = self.equalized_lr ),
      self.nl
    )

  # def get_fmap( self, scale_stage ):
  #   return min( int( FMAP_BASE / ( 2**scale_stage ) ), FMAP_MAX )

  def get_mbstd_layer( self ):
    if self.mbstd_group_size == -1:
      mbstd_layer = []
    else:
      mbstd_layer = [ Lambda(
        lambda x, group_size: concat_mbstd_layer( x, group_size ),
        group_size = self.mbstd_group_size
      ) ]

    return mbstd_layer

  # NOTE: Perhaps remove for loop and ideally just make it all one function (i.e. `return self.func( x )`)
  def forward( self, x ):
    x = self.preprocess_x( x )
    if self.fade_in_phase:
      x = self.prev_fromrgb( self.pooler_skip_connection( x ) ) * ( 1. - self.alpha ) + \
          self.disc_blocks[ 0 ]( self.fromrgb( x ) ) * ( self.alpha )
    else:
      x = self.disc_blocks[ 0 ]( self.fromrgb( x ) )
    for disc_block in self.disc_blocks[ 1: ]:
      x = disc_block( x )
    return x.view( -1 )