# -*- coding: UTF-8 -*-

"""Learner for StyleGANs.

  Typical usage example:

  First configure your desired GAN on the command-line:
    go to root directory...
    $ python config.py stylegan
    $ python data_config.py FFHQ path/to/datasets/ffhq

  Then write a custom script (or use train.py):
    from gan_zoo import get_current_configuration
    from gan_zoo.utils.data_utils import prepare_dataset, prepare_dataloader
    from gan_zoo.stylegan.learner import StyleGANLearner

    # get most recent configurations:
    config = get_current_configuration( 'config' )
    data_config = get_current_configuration( 'data_config' )

    # get DataLoader(s)
    train_ds, valid_ds = prepare_dataset( data_config )
    train_dl, valid_dl, z_valid_dl = prepare_dataloader( config, data_config, train_ds, valid_ds )

    # instantiate StyleGANLearner and train:
    learner = StyleGANLearner( config )
    learner.train( train_dl, valid_dl, z_valid_dl )   # train for config.num_main_iters iterations
    learner.config.num_main_iters = 300000            # this is one example of changing your instantiated learner's configurations
    learner.train( train_dl, valid_dl, z_valid_dl )   # train for another 300000 iterations

Note that the above custom script is just a more flexible alternative to running
train.py (you can, for example, run the above on a Jupyter Notebook). You can
always just run train.py.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from .base import StyleGAN
from .architectures import StyleAddNoise, StyleGenerator
from _int import get_current_configuration, LearnerConfigCopy
from progan.architectures import ProDiscriminator
from progan.learner import ProGANLearner
from utils.latent_utils import gen_rand_latent_vars
from utils.backprop_utils import calc_gp, configure_adam_for_gan
from utils.custom_layers import Conv2dBias, LinearBias

from abc import ABC
import copy
import logging
from pathlib import Path
from functools import partial
from itertools import accumulate
from timeit import default_timer as timer

import numpy as np
from PIL import Image
from indexed import IndexedOrderedDict
import matplotlib.pyplot as plt
plt.rcParams.update( { 'figure.max_open_warning': 0 } )
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

# from tqdm import tqdm

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

NONREDEFINABLE_ATTRS = ( 'model', 'init_res', 'res_samples', 'res_dataset', 'len_latent',
                         'num_classes', 'class_condition', 'use_auxiliary_classifier',
                         'model_upsample_type', 'model_downsample_type', 'align_corners',
                         'blur_type', 'nonlinearity', 'use_equalized_lr', 'normalize_z',
                         'use_pixelnorm', 'mbstd_group_size', 'use_ewma_gen',
                         'use_instancenorm', 'use_noise', 'pct_mixing_reg',
                         'beta_trunc_trick', 'psi_trunc_trick', 'cutoff_trunc_trick',
                         'len_dlatent', 'mapping_num_fcs', 'mapping_lrmul', )

REDEFINABLE_FROM_LEARNER_ATTRS = ( 'batch_size', 'loss', 'gradient_penalty',
                                   'optimizer', 'lr_sched', 'latent_distribution', )

COMPUTE_EWMA_VIA_HALFLIFE = True
EWMA_SMOOTHING_HALFLIFE = 10.
EWMA_SMOOTHING_BETA = .999

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class StyleGANLearner( ProGANLearner ):
  """GAN Learner specifically designed for StyleGAN architectures.

  Once instantiated, the StyleGANLearner object's configuration can be changed, but only via
  its self.config attribute (i.e. running 'python config.py [model]' post-instantiation
  will not affect this learner's configuration).
  """
  def __init__( self, config ):
    super( StyleGANLearner, self ).__init__( config )

    if self.model == 'StyleGAN':

      # If you want to change an attribute in an already-instantiated StyleGANLearner's config or data_config,
      # change self.config (below) instead of config and self.data_config (also below) instead of data_config:
      self.config = LearnerConfigCopy( config,
                                       self.__class__.__name__,
                                       NONREDEFINABLE_ATTRS,
                                       REDEFINABLE_FROM_LEARNER_ATTRS )
      # pretrained models loaded later on for evaluation should not require data_config.py to have been run:
      self._get_data_config( raise_exception = False )

      self._latent_distribution = self.config.latent_distribution

      global StyleGAN, StyleGenerator, ProDiscriminator, StyleDiscriminator
      StyleGAN = type( 'StyleGAN', ( nn.Module, ABC, ), dict( StyleGAN.__dict__ ) )
      StyleGAN.reset_state( )

      StyleGenerator = type( 'StyleGenerator', ( StyleGAN, ), dict( StyleGenerator.__dict__ ) )
      self.gen_model = StyleGenerator(
        final_res = self.config.res_samples,
        latent_distribution = self._latent_distribution,
        len_latent = self.config.len_latent,
        len_dlatent = self.config.len_dlatent,
        mapping_num_fcs = self.config.mapping_num_fcs,
        mapping_lrmul = self.config.mapping_lrmul,
        use_instancenorm = self.config.use_instancenorm,
        use_noise = self.config.use_noise,
        upsampler = self.gen_model_upsampler,
        blur_type = self.config.blur_type,
        nl = self.nl,
        num_classes = self.num_classes_gen,
        equalized_lr = self.config.use_equalized_lr,
        normalize_z = self.config.normalize_z,
        use_pixelnorm = self.config.use_pixelnorm,
        pct_mixing_reg = self.config.pct_mixing_reg,
        truncation_trick_params = { 'beta': self.config.beta_trunc_trick,
                                    'psi': self.config.psi_trunc_trick,
                                    'cutoff_stage': self.config.cutoff_trunc_trick }
      )

      # Create `StyleDiscriminator` type object by converting `ProDiscriminator` class into `StyleDiscriminator` class:
      StyleDiscriminator = type( 'StyleDiscriminator', ( StyleGAN, ), dict( ProDiscriminator.__dict__ ) )
      self.disc_model = StyleDiscriminator(
        final_res = self.config.res_samples,
        pooler = self.disc_model_downsampler,
        blur_type = self.config.blur_type,
        nl = self.nl,
        num_classes = self.num_classes_disc,
        equalized_lr = self.config.use_equalized_lr,
        mbstd_group_size = self.config.mbstd_group_size
      )

      # If one wants to start at a higher resolution than 4:
      assert self.config.init_res <= self.config.res_samples
      if self.config.init_res > 4:
        _init_res_log2 = int( np.log2( self.config.init_res ) )
        if float( self.config.init_res ) != 2**_init_res_log2:
          raise ValueError( 'Only resolutions that are powers of 2 are supported.' )
        num_scale_inc = _init_res_log2 - 2
        for _ in range( num_scale_inc ):
          self.gen_model.increase_scale()
          self.disc_model.increase_scale()
        self.gen_model.fade_in_phase = False  # this applies it to both networks simultaneously

      # Generator and Discriminator state data must match:
      assert self.gen_model.cls_base.__dict__ == \
             self.disc_model.cls_base.__dict__

      # Initialize EWMA Generator Model:
      self.gen_model_lagged = None
      if self.config.use_ewma_gen:
        _orig_mode = self.gen_model.training
        self.gen_model.train()
        self.gen_model.to( 'cpu' )
        with torch.no_grad():
          self.gen_model_lagged = copy.deepcopy( self.gen_model )  # for memory efficiency in GPU
          self.gen_model_lagged.to( self.config.metrics_dev )
          self.gen_model_lagged.train( mode = _orig_mode )
        self.gen_model.train( mode = _orig_mode )

      self.gen_model.to( self.config.dev )
      self.disc_model.to( self.config.dev )

      self.batch_size = self.config.bs_dict[ self.gen_model.curr_res ]

      if self.cond_gen:
        self.labels_one_hot_disc = self._tensor( self.batch_size, self.num_classes )
        self.labels_one_hot_gen = self._tensor( self.batch_size * self.config.gen_bs_mult, self.num_classes )

      # Loss Function:
      self._loss = config.loss.casefold()
      self._set_loss( )

      # Gradient Regularizer:
      self.gp_func = partial(
        calc_gp,
        gp_type = self._gradient_penalty,
        nn_disc = self.disc_model,
        lda = self.config.lda,
        gamma = self.config.gamma
      )

      # Optimizer:
      self._set_optimizer( )

      # Epsilon Loss to punish possible outliers from training distribution:
      self.eps = False
      if self.config.eps_drift > 0:
        self.eps = True

      # Print configuration:
      print( '------------- Training Configuration ------------' )
      for k, v in vars( config ).items():
        print( f'  {k}: {v}' )
      print( '-------------------------------------------------' )
      # print( "  If you would like to alter any of the above configurations,\n" + \
      #        "  please do so via altering your instantiated StyleGANLearner().config's attributes." )
      print( '\n    Ready to train!\n' )

  def apply_lagged_weights( self, m ):
    # TODO: Include support for other learnable layers such as BatchNorm
    _keys = m.state_dict().keys()
    if isinstance( m, ( nn.Linear, nn.Conv2d, LinearBias, Conv2dBias, ) ):
      if 'weight' in _keys:
        m.weight = nn.Parameter( self.lagged_params.values()[ self._param_tensor_num + 1 ] )
        self._param_tensor_num += 1
      if 'bias' in _keys:
        m.bias = nn.Parameter( self.lagged_params.values()[ self._param_tensor_num + 1 ] )
        self._param_tensor_num += 1
    elif isinstance( m, StyleAddNoise ):
      if 'noise_weight' in _keys:
        m.noise_weight = nn.Parameter( self.lagged_params.values()[ self._param_tensor_num + 1 ] )
        self._param_tensor_num += 1
    elif isinstance( m, StyleGenerator ):
      if 'const_input' in _keys:
        m.const_input = nn.Parameter( self.lagged_params.values()[ 0 ] )
        self._param_tensor_num += 1

  @property
  def latent_distribution( self ):
    return self._latent_distribution

  @latent_distribution.setter
  def latent_distribution( self, new_latent_distribution ):
    new_latent_distribution = new_latent_distribution.casefold()
    self._latent_distribution = new_latent_distribution
    self.gen_model.latent_distribution = new_latent_distribution

  # def reset_stylegan_state( self ):
  #   self.gen_model.cls_base.reset_state( )  # this applies to both networks simultaneously

  # .......................................................................... #

  @torch.no_grad()
  def plot_sample( self, z_test, z_mixing = None, style_mixing_stage = None, noise = None, label = None, time_average = True ):
    """Plots and shows 1 sample from input latent code; offers stylemixing and noise input capabilities."""
    if not self.pretrained_model:
      if not self._is_data_configed:
        self._get_data_config( raise_exception = True )

    for z in ( z_test, z_mixing, ):
      if z is not None:
        if z.dim() == 2:
          if z.shape[0] != 1:
            raise IndexError( 'This method only permits plotting 1 generated sample at a time.' )
        elif z.dim() != 1:
          raise IndexError( 'Incorrect dimensions of input latent vector. Must be either `dim == 1` or `dim == 2`.' )
        if not self.cond_gen:
          if z.shape[-1] != self.config.len_latent:
            message = f"Input latent vector must be of size {self.config.len_latent}."
            raise IndexError( message )
        else:
          if z.shape[-1] != self.config.len_latent + self.num_classes_gen:
            message = f"This is a generator class-conditioned model. So please make sure to append a one-hot encoded vector\n" + \
                      f"of size {self.num_classes_gen} that indicates the to-be generated sample's class to a latent vector of\n" + \
                      f"size {self.config.len_latent}. Total input size must therefore be {self.config.len_latent + self.num_classes_gen}."
            raise IndexError( message )

    # z_test = z_test.to( self.config.dev )
    x_test = \
      self.gen_model_lagged(
        z_test,
        x_mixing = z_mixing,
        style_mixing_stage = style_mixing_stage,
        noise = noise ).squeeze() if time_average else \
      self.gen_model(
        z_test,
        x_mixing = z_mixing,
        style_mixing_stage = style_mixing_stage,
        noise = noise ).squeeze()

    if label is not None:
      print( f'Label Index for Generated Image: {label}' )

    logger = logging.getLogger()
    _old_level = logger.level
    logger.setLevel( 100 )  # ignores potential "clipping input data" warning
    plt.imshow( ( ( ( x_test ) \
                          .cpu().detach() * self._ds_std ) + self._ds_mean ) \
                          .numpy().transpose( 1, 2, 0 ), interpolation = 'none'
    )
    logger.setLevel( _old_level )
    plt.show()

  @torch.no_grad()
  def make_stylemixing_grid( self, zs_sourceb, zs_coarse = [], zs_middle = [], zs_fine = [], labels = None, time_average = True ):
    """Generates style-mixed grid of images, emulating Figure 3 in Karras et al. 2019."""
    assert any( len( zs ) for zs in ( zs_coarse, zs_middle, zs_fine, ) )
    if not self.pretrained_model:
      if not self._is_data_configed:
        self._get_data_config( raise_exception = True )

    szs = []
    stages = [ 0 ]
    for m, zs in enumerate( ( zs_sourceb, zs_coarse, zs_middle, zs_fine, ) ):
      szs.append( 1 if isinstance( zs, torch.Tensor ) and zs.dim() == 1 else len( zs ) )
      if szs[m]:
        if zs.dim() > 2:
          raise IndexError( 'Incorrect dimensions of input latent vector. Must be either `dim == 1` or `dim == 2`.' )
        else:
          if m == 1:
            stage = 1  # stage for style-mixing of coarse features from source b
            if zs.dim() == 1: zs_coarse.unsqueeze_( dim = 0 )
          elif m == 2:
            stage = 4  # stage for style-mixing of mid-level features from source b
            if zs.dim() == 1: zs_middle.unsqueeze_( dim = 0 )
          elif m == 3:
            stage = 8  # stage for style-mixing of fine-grained features from source b
            if zs.dim() == 1: zs_fine.unsqueeze_( dim = 0 )
          if m: stages.append( stage )
        if not self.cond_gen:
          if zs.shape[1] != self.config.len_latent:
            message = f"Input latent vector must be of size {self.config.len_latent}."
            raise IndexError( message )
        else:
          if zs.shape[1] != self.config.len_latent + self.num_classes_gen:
            message = f"This is a generator class-conditioned model. So please make sure to append a one-hot encoded vector\n" + \
                      f"of size {self.num_classes_gen} that indicates the to-be generated sample's class to a latent vector of\n" + \
                      f"size {self.config.len_latent}. Total input size must therefore be {self.config.len_latent + self.num_classes_gen}."
            raise IndexError( message )
      else:
          stages.append( 0 )

    ncols_tot = 1 + szs[0]; nrows_tot = 1 + sum( szs[1:] )
    cum_szs = [0] + list( accumulate( szs[1:] ) )
    fig = plt.figure( figsize = ( 8. * ( ncols_tot / nrows_tot ), 8. if labels is None else 9., ) )
    axs = fig.subplots( ncols = ncols_tot, nrows = nrows_tot )
    fig.tight_layout( pad = 0 )
    _fctrs = ( ( fig.subplotpars.wspace / 0.47267497603068315 ), ( fig.subplotpars.hspace / 0.28000000000000086 ), )
    gaps = ( ( .002 * _fctrs[0] * ( fig.dpi / fig.get_size_inches().mean() ) ), \
              ( .002 * _fctrs[1] * ( fig.dpi / fig.get_size_inches()[1] ) ), )
    # wspace = fig.subplotpars.wspace
    # hspace = fig.subplotpars.hspace
    wspace = plt.rcParams[ 'figure.subplot.wspace' ] * _fctrs[0]
    hspace = plt.rcParams[ 'figure.subplot.hspace' ] * _fctrs[1]
    start = 0
    logger = logging.getLogger(); _old_level = logger.level; logger.setLevel( 100 )  # ignores potential "clipping input data" warning
    for m, zs in enumerate( ( zs_sourceb, zs_coarse, zs_middle, zs_fine, ) ):
      if szs[m]:
        nrows = 1 if not m else szs[ m ]
        if m: start = row + 1
        for row in range( start, start + nrows ):
          for col in range( ncols_tot ):
            axs[row][col].axis( 'off' )
            axs[row][col].set_aspect( 'equal' )
            l, b, w, h = axs[row][col].get_position().bounds
            l *= ( 1. - wspace ); b *= ( 1. - hspace ); w *= ( 1. + wspace ); h *= ( 1. + hspace )
            if m > 1: b -= ( m - 1 ) *gaps[1]
            if not row and not col:
              axs[row][col].set_position( [ l - gaps[0], b + gaps[1], w, h ] )
            else:
              if ( not row ) != ( not col ):  # xor
                if row:
                  axs[row][col].set_position( [ l - gaps[0], b, w, h ] )
                  if row == 1:
                    axs[row][col].set_title( 'Source A', fontweight = 'bold' )
                elif col:
                  axs[row][col].set_position( [ l, b + gaps[1], w, h ] )
                  if col == 1:
                    axs[row][col].axis( 'on' )
                    axs[row][col].spines['top'].set_visible(False)
                    axs[row][col].spines['right'].set_visible(False)
                    axs[row][col].spines['bottom'].set_visible(False)
                    axs[row][col].spines['left'].set_visible(False)
                    axs[row][col].set_xticks( [] )
                    axs[row][col].set_yticks( [] )
                    axs[row][col].set_ylabel( 'Source B', fontweight = 'bold', fontsize = plt.rcParams[ 'axes.titlesize' ] )
                x = self.gen_model_lagged( zs_sourceb[ col - 1 ] if not m else zs[ row - cum_szs[m-1] - 1 ] ).squeeze() if time_average else \
                    self.gen_model( zs_sourceb[ col - 1 ] if not m else zs[ row - cum_szs[m-1] - 1 ] ).squeeze()
              elif row and col:
                axs[row][col].set_position( [ l, b, w, h ] )
                x = self.gen_model_lagged( zs[ row - cum_szs[m-1] - 1 ], x_mixing = zs_sourceb[ col - 1 ], style_mixing_stage = stages[ m ] ).squeeze() if time_average else \
                    self.gen_model( zs[ row - cum_szs[m-1] - 1 ], x_mixing = zs_sourceb[ col - 1 ], style_mixing_stage = stages[ m ] ).squeeze()
              axs[row][col].imshow(
                ( ( x.cpu().detach() * self._ds_std ) + self._ds_mean ).numpy().transpose( 1, 2, 0 ), interpolation = 'none'
              )
              if labels is not None:
                axs[row][col].set_title( str( labels[ row*ncols_tot + col ].item() ) )

    # center the axes of each subplot on the figure after moving everything around
    fig_top_right = fig.get_window_extent().corners()[3]
    l,b,w,h = axs[0,-1].get_position().bounds; top_right = ( l + w, b + h, )
    l,b,w,h = axs[-1,0].get_position().bounds; bot_left = ( l, b, )
    shift = tuple( -bot_left[i] + ( 1. - top_right[i] ) / 2. for i in range( 2 ) )
    for row in range( nrows_tot ):
      for col in range( ncols_tot ):
        l, b, w, h = axs[row][col].get_position().bounds
        axs[row][col].set_position( [ l + shift[0], b + shift[1], w, h ] )

    # ylabel for Coarse, Middle, and Fine styles from Source B
    for k in range( 1, m + 1 ):
      if szs[k]:
        l, b, w, h = axs[cum_szs[k]][0].get_position().bounds
        center_y = ( szs[k] * h ) / 2.
        if k == 1: ylabel = 'Coarse'
        elif k == 2: ylabel = 'Middle'
        elif k == 3: ylabel = 'Fine'
        ylabel += ' from B' if szs[k] < 2 else ' styles from Source B'
        fig.text( l - gaps[0], b + center_y, ylabel, va = 'center', rotation = 'vertical' )
    logger.setLevel( _old_level )

    return ( fig, axs, )