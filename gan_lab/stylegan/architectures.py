# -*- coding: UTF-8 -*-

"""StyleGAN architectures.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from .base import StyleGAN
from _int import FMAP_SAMPLES, RES_INIT
from utils.latent_utils import gen_rand_latent_vars
from utils.custom_layers import Lambda, get_blur_op, NormalizeLayer, \
                                Conv2dEx, LinearEx, Conv2dBias

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

FMAP_G_INIT_FCTR = 1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class StyleMappingNetwork( nn.Module ):
  """Mapping Network for StyleGAN architecture."""
  def __init__( self,
                len_latent = 512,
                len_dlatent = 512,
                num_fcs = 8,
                lrmul = .01,
                nl = nn.LeakyReLU( negative_slope = .2 ),
                equalized_lr = True,
                normalize_z = True ):
    super( StyleMappingNetwork, self ).__init__()

    if normalize_z:
      self.preprocess_z = nn.Sequential(
        Lambda( lambda x: x.view( -1, len_latent ) ),
        NormalizeLayer( 'PixelNorm' )
      )
    else:
      self.preprocess_z = Lambda( lambda x: x.view( -1, len_latent ) )

    self.dims = np.linspace( len_latent, len_dlatent, num_fcs + 1 ).astype( np.int64 )
    self.fc_mapping_model = nn.Sequential( )
    for seq_n in range( num_fcs ):
      self.fc_mapping_model.add_module(
        'fc_' + str( seq_n ),
        LinearEx( nin_feat = self.dims[seq_n], nout_feat = self.dims[seq_n+1],
                  init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                  equalized_lr = equalized_lr, lrmul = lrmul )
      )
      self.fc_mapping_model.add_module( 'nl_' + str( seq_n ), nl )

  def forward( self, x ):
    return self.fc_mapping_model( self.preprocess_z( x ) )


class StyleConditionedMappingNetwork( nn.Module ):
  """Class-conditioned version of Mapping Network for StyleGAN architecture."""
  def __init__( self,
                num_classes,
                len_latent = 512,
                len_dlatent = 512,
                num_fcs = 8,
                lrmul = .01,
                nl = nn.LeakyReLU( negative_slope = .2 ),
                equalized_lr = True,
                normalize_z = True,
                embed_cond_vars = True ):
    super( StyleConditionedMappingNetwork, self ).__init__()

    self.len_latent = len_latent
    self.num_classes = num_classes

    self.embed_cond_vars = embed_cond_vars
    if embed_cond_vars:
      self.class_embedding = LinearEx( nin_feat = num_classes, nout_feat = len_latent,
                                       init = None, init_type = 'Standard Normal', include_bias = False )

    self.dims = np.linspace( len_latent, len_dlatent, num_fcs ).astype( np.int64 )
    self.dims = np.insert( self.dims, 0, 2*len_latent if self.embed_cond_vars else len_latent + num_classes )
    self.fc_mapping_model = nn.Sequential( )
    if normalize_z:
      self.fc_mapping_model.add_module( 'pixelnorm', NormalizeLayer( 'PixelNorm' ) )
    for seq_n in range( num_fcs ):
      self.fc_mapping_model.add_module(
        'fc_' + str( seq_n ),
        LinearEx( nin_feat = self.dims[seq_n], nout_feat = self.dims[seq_n+1],
                  init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                  equalized_lr = equalized_lr, lrmul = lrmul )
      )
      self.fc_mapping_model.add_module( 'nl_' + str( seq_n ), nl )

  def forward( self, x, y ):
    y = y.view( -1, self.num_classes )
    if self.embed_cond_vars:
      y = self.class_embedding( y )
    return self.fc_mapping_model( torch.cat( ( x.view( -1, self.len_latent ), y, ), dim = 1 ) )


class StyleAddNoise( nn.Module ):
  """Simple `nn.Module` that adds weighted uncorrelated Gaussian noise to a layer of feature maps."""
  def __init__( self, nf ):
    super( StyleAddNoise, self ).__init__()

    self.noise_weight = nn.Parameter( torch.FloatTensor( 1, nf, 1, 1 ).fill_( 0 ) )

  def forward( self, x, noise = None ):
    if self.training or noise is None:
      # for training mode or when user does not supply noise in evaluation mode:
      return x + self.noise_weight * \
        torch.randn( x.shape[0], 1, x.shape[2], x.shape[3], dtype = torch.float32, device = x.device )
    else:
      # for if when user supplies noise in evaluation mode:
      return x + self.noise_weight * noise

# ............................................................................ #
# Generator:
# ----------

# TODO: Implement more efficient "recursive structure" by Tero Karras for training (see their GitHub implementation)
class StyleGenerator( StyleGAN ):
  """StyleGAN (Karras et al. 2019) Generator

  Emulates most recent official implementation.
  """
  def __init__( self,
                final_res,
                latent_distribution = 'normal',
                len_latent = 512,
                len_dlatent = 512,
                mapping_num_fcs = 8,
                mapping_lrmul = .01,
                use_instancenorm = True,
                use_noise = True,
                upsampler = nn.Upsample( scale_factor = 2, mode = 'nearest' ),
                blur_type = None,
                nl = nn.LeakyReLU( negative_slope = .2 ),
                num_classes = 0,
                equalized_lr = True,
                normalize_z = True,
                use_pixelnorm = False,
                pct_mixing_reg = .9,
                truncation_trick_params = { 'beta': .995, 'psi': .7, 'cutoff_stage': 4 } ):

    super( self.__class__, self ).__init__( final_res )

    self.gen_layers = nn.ModuleList( )

    self.upsampler = upsampler
    self.upsampler_skip_connection = \
      lambda xb: F.interpolate( xb, scale_factor = 2, mode = 'nearest' )  # keep fading-in layers simple

    self.gen_blur_type = blur_type

    self.nl = nl

    self.equalized_lr = equalized_lr

    self.pct_mixing_reg = pct_mixing_reg
    self._use_mixing_reg = True if pct_mixing_reg else False

    self.latent_distribution = latent_distribution
    self.len_latent = len_latent
    self.len_dlatent = len_dlatent
    assert isinstance( num_classes, int )
    self.num_classes = num_classes

    # Mapping Network initialization:
    if not num_classes:
      self.z_to_w = StyleMappingNetwork(
        len_latent = len_latent,
        len_dlatent = len_dlatent,
        num_fcs = mapping_num_fcs,
        lrmul = mapping_lrmul,
        nl = nl,
        equalized_lr = equalized_lr,
        normalize_z = normalize_z
      )
    else:
      self.z_to_w = StyleConditionedMappingNetwork(
        num_classes,
        len_latent = len_latent,
        len_dlatent = len_dlatent,
        num_fcs = mapping_num_fcs,
        lrmul = mapping_lrmul,
        nl = nl,
        equalized_lr = equalized_lr,
        normalize_z = normalize_z
      )

    _fmap_init = len_latent * FMAP_G_INIT_FCTR

    # initializing the input to 1 has about the same effect as applyng PixelNorm to the input
    self.const_input = nn.Parameter(
      torch.FloatTensor( 1, _fmap_init, RES_INIT, RES_INIT ).fill_( 1 )
    )

    self._use_noise = use_noise
    self._trained_with_noise = use_noise
    if use_noise:
      conv = Conv2dEx( ni = _fmap_init, nf = self.fmap, ks = 3,
                       stride = 1, padding = 1, init = 'He', init_type = 'StyleGAN',
                       gain_sq_base = 2., equalized_lr = equalized_lr, include_bias = False )
      noise = [
        StyleAddNoise( nf = _fmap_init ),
        StyleAddNoise( nf = self.fmap ),
      ]
      bias = (
        [ Conv2dBias( nf = _fmap_init ) ],
        [ Conv2dBias( nf = self.fmap ) ],
      )
    else:
      conv = Conv2dEx( ni = _fmap_init, nf = self.fmap, ks = 3,
                       stride = 1, padding = 1, init = 'He', init_type = 'StyleGAN',
                       gain_sq_base = 2., equalized_lr = equalized_lr, include_bias = True )
      # noise = ( [], [], )
      noise = [ None, None ]
      bias = ( [], [], )  # NOTE: without noise, the bias would get directly added to the constant input, so the constant input can just learn this bias,
                          #       so theoretically, there shouldn't be a need to include the bias either. There may be numerical approximation problems from backprop, however.

    norms = []
    self.use_pixelnorm = use_pixelnorm
    if use_pixelnorm:
      norms.append( NormalizeLayer( 'PixelNorm' ) )
    self.use_instancenorm = use_instancenorm
    if use_instancenorm:
      norms.append( NormalizeLayer( 'InstanceNorm' ) )

    w_to_styles = (
      LinearEx( nin_feat = self.z_to_w.dims[ -1 ], nout_feat = 2 * _fmap_init,
                init = 'He', init_type = 'StyleGAN', gain_sq_base = 1., equalized_lr = equalized_lr ),
      LinearEx( nin_feat = self.z_to_w.dims[ -1 ], nout_feat = 2 * self.fmap,
                init = 'He', init_type = 'StyleGAN', gain_sq_base = 1., equalized_lr = equalized_lr ),
    )
    assert 0. <= truncation_trick_params[ 'beta' ] <= 1.
    self.w_ewma_beta = truncation_trick_params[ 'beta' ]
    self._w_eval_psi = truncation_trick_params[ 'psi' ]  # allow psi to be any number you want, perhaps worthy of experimentation
    assert ( ( isinstance( truncation_trick_params[ 'cutoff_stage' ], int ) and \
                0 < truncation_trick_params[ 'cutoff_stage' ] <= int( np.log2( self.final_res ) ) - 2 ) or \
                truncation_trick_params[ 'cutoff_stage' ] is None )
    self._trunc_cutoff_stage = truncation_trick_params[ 'cutoff_stage' ]
    # set the below to `False` if you want to turn off during evaluation mode
    self.use_truncation_trick = True if self._trunc_cutoff_stage else False
    self.w_ewma = None

    self.gen_layers.append(
      nn.ModuleList( [
        None,
        noise[0],
        nn.Sequential( *bias[0], nl, *norms ),
        w_to_styles[0]
      ] )
    )
    self.gen_layers.append(
      nn.ModuleList( [
        conv,
        noise[1],
        nn.Sequential( *bias[1], nl, *norms ),
        w_to_styles[1]
      ] )
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

    self.gen_layers.append(
      self.get_conv_layer( ni = self.fmap_prev, upsample = True, blur_op = blur_op )
    )
    self.gen_layers.append(
      self.get_conv_layer( ni = self.fmap )
    )

    self.prev_torgb = copy.deepcopy( self.torgb )
    self._update_torgb( ni = self.fmap )

  def get_conv_layer( self, ni, upsample = False, blur_op = None, append_nl = True ):
    upsampler = []
    if upsample:
      upsampler.append( self.upsampler )

    if self.use_noise or blur_op is not None:
      conv = Conv2dEx( ni = ni, nf = self.fmap, ks = 3, stride = 1, padding = 1,
                       init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                       equalized_lr = self.equalized_lr, include_bias = False )
      bias = [ Conv2dBias( nf = self.fmap ) ]
    else:
      conv = Conv2dEx( ni = ni, nf = self.fmap, ks = 3, stride = 1, padding = 1,
                       init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                       equalized_lr = self.equalized_lr, include_bias = True )
      bias = []

    blur = []
    if blur_op is not None:
      assert isinstance( blur_op, nn.Module )
      blur.append( blur_op )

    noise = None
    if self.use_noise:
      noise = StyleAddNoise( nf = self.fmap )

    nl = []
    if append_nl:
      nl.append( self.nl )

    norms = []
    if self.use_pixelnorm:
      norms.append( NormalizeLayer( 'PixelNorm' ) )
    if self.use_instancenorm:
      norms.append( NormalizeLayer( 'InstanceNorm' ) )

    w_to_style = LinearEx( nin_feat = self.z_to_w.dims[ -1 ], nout_feat = 2*self.fmap,
                           init = 'He', init_type = 'StyleGAN', gain_sq_base = 1.,
                           equalized_lr = self.equalized_lr )

    return nn.ModuleList( [ nn.Sequential( *upsampler, conv, *blur ),
                            noise,
                            nn.Sequential( *( bias + nl + norms ) ),
                            w_to_style ] )

    # return nn.ModuleList( [ nn.Sequential( *upsampler, conv, *( blur + noise + bias + nl + norms ) ), w_to_style ] )

  def _update_torgb( self, ni ):
    self.torgb = Conv2dEx( ni = ni, nf = FMAP_SAMPLES, ks = 1, stride = 1,
                           padding = 0, init = 'He', init_type = 'StyleGAN',
                           gain_sq_base = 1., equalized_lr = self.equalized_lr )

  def train( self, mode = True ):
    """Overwritten to turn on mixing regularization (if > 0%) during training mode."""
    super( self.__class__, self ).train( mode = mode )

    self._use_noise = self._trained_with_noise
    self._use_mixing_reg = True if self.pct_mixing_reg else False

  def eval( self ):
    """Overwritten to turn off mixing regularization during evaluation mode."""
    super( self.__class__, self ).eval( )

    self._use_mixing_reg = False

  def to( self, *args, **kwargs ):
    """Overwritten to allow for non-Parameter objects' Tensors to be sent to the appropriate device."""
    super( self.__class__, self ).to( *args, **kwargs )

    for arg in args:
      if arg in ( 'cpu', 'cuda', ) or isinstance( arg, torch.device ):
        if self.w_ewma is not None:
          self.w_ewma = self.w_ewma.to( arg )
          break

  @property
  def use_noise( self ):
    return self._use_noise

  @use_noise.setter
  def use_noise( self, mode ):
    """Allows for optionally evaluating without noise inputs."""
    if self.training:
      raise Exception( 'Once use_noise argument is set, it cannot be changed' + \
                       ' for training purposes. It can, however, be changed in eval mode.' )
    elif not self._trained_with_noise:
      raise Exception( 'Model was not trained with noise, so cannot use noise in eval mode.' )
    else:
      self._use_noise = mode

  @property
  def w_eval_psi( self ):
    return self._w_eval_psi

  @w_eval_psi.setter
  def w_eval_psi( self, new_w_eval_psi ):
    """Change this to your choosing (but only in evaluation mode), optionally allowing for |psi| to be > 1."""
    if not self.training:
      self._w_eval_psi = new_w_eval_psi
    else:
      raise Exception( 'Can only alter psi value for truncation trick on w during evaluation mode.' )

  @property
  def trunc_cutoff_stage( self ):
    return self._trunc_cutoff_stage

  @trunc_cutoff_stage.setter
  def trunc_cutoff_stage( self, new_trunc_cutoff_stage ):
    """Change this to your choosing (but only in evaluation mode)."""
    if not self.training:
      _final_stage = int( np.log2( self.final_res ) ) - 1
      if ( isinstance( new_trunc_cutoff_stage, int ) and \
        0 < new_trunc_cutoff_stage <= _final_stage ) or new_trunc_cutoff_stage is None:
        self._trunc_cutoff_stage = new_trunc_cutoff_stage
      else:
        message = f'Input cutoff stage for truncation trick on w must be of type `int` in range (0,{_final_stage}] or `None`.'
        raise ValueError( message )
    else:
      raise Exception( 'Can only alter cutoff stage for truncation trick on w during evaluation mode.' )

  def forward( self, x, x_mixing = None, style_mixing_stage:int = None, noise = None ):
    # TODO: Implement the ability to style-mix more than just 2 styles in eval mode
    # TODO: Implement the ability to input the disentangled latent variable w directly

    cutoff_idx = None
    # Training Mode Only:
    if self._use_mixing_reg:
      if np.random.rand() < self.pct_mixing_reg:
        if self.alpha != 0:
          cutoff_idx = torch.randint( 1, 2*self.scale_stage, ( 1, ) ).item()
        else:
          cutoff_idx = torch.randint( 1, 2*self.scale_stage - 2, ( 1, ) ).item()

    x = self.z_to_w( x )
    bs = x.shape[0]

    if self.use_truncation_trick:
      # Training Mode Only:
      if self.training:
        if self.w_ewma is None:
          self.w_ewma = x.detach().clone().mean( dim = 0 )
        else:
          with torch.no_grad():
            # TODO: Implement a memory-efficient method to compute this for the ewma generator
            #       (currently just using the same average w for the generator and the ewma generator)
            self.w_ewma = x.mean( dim = 0 ) * ( 1. - self.w_ewma_beta ) + \
                          self.w_ewma * ( self.w_ewma_beta )
      # Evaluation Mode Only:
      elif self.trunc_cutoff_stage is not None:
        x = self.w_ewma.expand_as( x ) + self.w_eval_psi * ( x - self.w_ewma.expand_as( x ) )

    out = self.const_input.expand( bs, -1, -1, -1 )

    if self.fade_in_phase:
      for n, layer in enumerate( self.gen_layers[ :-2 ] ):
        if n:
          out = layer[ 0 ]( out )
        if self.use_noise:
          out = layer[ 1 ]( out, noise = noise[ n ] if noise is not None else None )
        out = layer[ 2 ]( out )

        if n == cutoff_idx:
          # TODO: Implement embedding-style conditioning from "Which Training Methods for
          #       GANs do actually Converge" & discriminator conditioning.
          x = gen_rand_latent_vars( num_samples = bs, length = self.len_latent,
                                    distribution = self.latent_distribution, device = x.device )
          x.requires_grad_( True )
          x = self.z_to_w( x )

        y = layer[ 3 ]( x ).view( -1, 2, layer[ 3 ].nout_feat // 2, 1, 1 )
        out = out * ( y[ :, 0 ].contiguous().add( 1 ) ) + \
              y[ :, 1 ].contiguous()  # add 1 for skip-connection effect

      # TODO: there should be a cleaner way to do the fading-in part while remaining memory-efficient...
      n += 1
      if n == cutoff_idx:
        x = gen_rand_latent_vars( num_samples = bs, length = self.len_latent,
                                  distribution = self.latent_distribution, device = x.device )
        x.requires_grad_( True )
        x = self.z_to_w( x )
      y = self.gen_layers[ -2 ][ 3 ]( x ).view( -1, 2, self.gen_layers[ -2 ][ 3 ].nout_feat // 2, 1, 1 )

      n += 1
      if n == cutoff_idx:
        x = gen_rand_latent_vars( num_samples = bs, length = self.len_latent,
                                  distribution = self.latent_distribution, device = x.device )
        x.requires_grad_( True )
        x = self.z_to_w( x )
      yf = self.gen_layers[ -1 ][ 3 ]( x ).view( -1, 2, self.gen_layers[ -1 ][ 3 ].nout_feat // 2, 1, 1 )

      if self.use_noise:
        return self.upsampler_skip_connection( self.prev_torgb( out ) ) * ( 1. - self.alpha ) + \
          self.torgb(
            self.gen_layers[ -1 ][ 2 ]( self.gen_layers[ -1 ][ 1 ]( self.gen_layers[ -1 ][ 0 ](
              self.gen_layers[ -2 ][ 2 ]( self.gen_layers[ -2 ][ 1 ]( self.gen_layers[ -2 ][ 0 ]( out ), noise = noise[ -2 ] if noise is not None else None ) ) * ( y[ :, 0 ].contiguous().add( 1 ) ) + y[ :, 1 ].contiguous()
            ), noise = noise[ -1 ] if noise is not None else None ) ) * ( yf[ :, 0 ].contiguous().add( 1 ) ) + yf[ :, 1 ].contiguous()
          ) * ( self.alpha )
      else:
        return self.upsampler_skip_connection( self.prev_torgb( out ) ) * ( 1. - self.alpha ) + \
          self.torgb(
            self.gen_layers[ -1 ][ 2 ]( self.gen_layers[ -1 ][ 0 ](
              self.gen_layers[ -2 ][ 2 ]( self.gen_layers[ -2 ][ 0 ]( out ) ) * ( y[ :, 0 ].contiguous().add( 1 ) ) + y[ :, 1 ].contiguous()
            ) ) * ( yf[ :, 0 ].contiguous().add( 1 ) ) + yf[ :, 1 ].contiguous()
          ) * ( self.alpha )

    else:
      for n, layer in enumerate( self.gen_layers ):
        if n:
          out = layer[ 0 ]( out )
        if self.use_noise:
          out = layer[ 1 ]( out, noise = noise[ n ] if noise is not None else None )
        out = layer[ 2 ]( out )

        # Training Mode Only:
        if n == cutoff_idx:
          # TODO: Implement embedding-style conditioning from "Which Training Methods for
          #       GANs do actually Converge" & discriminator conditioning.
          x = gen_rand_latent_vars( num_samples = bs, length = self.len_latent,
                                    distribution = self.latent_distribution, device = x.device )
          x.requires_grad_( True )
          x = self.z_to_w( x )

        # Evaluation Mode Only:
        if n == style_mixing_stage:
          assert ( style_mixing_stage and not self.training and isinstance( x_mixing, torch.Tensor ) )
          x = self.z_to_w( x_mixing )
          # the new z that is sampled for style-mixing is already de-truncated
          if self.use_truncation_trick and self.trunc_cutoff_stage is not None and n < 2*self.trunc_cutoff_stage:
            x = self.w_ewma.expand_as( x ) + self.w_eval_psi * ( x - self.w_ewma.expand_as( x ) )
        elif self.use_truncation_trick and not self.training and self.trunc_cutoff_stage is not None and n == 2*self.trunc_cutoff_stage:
          # de-truncate w for higher resolutions; more memory-efficient than defining 2 w's
          x = ( x - self.w_ewma.expand_as( x ) ).div( self.w_eval_psi ) + self.w_ewma.expand_as( x )

        y = layer[ 3 ]( x ).view( -1, 2, layer[ 3 ].nout_feat // 2, 1, 1 )
        out = out * ( y[ :, 0 ].contiguous().add( 1 ) ) + \
              y[ :, 1 ].contiguous()  # add 1 for skip-connection effect

      return self.torgb( out )