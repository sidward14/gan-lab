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
from utils.backprop_utils import configure_adam_for_gan
from utils.custom_layers import Conv2dBias, LinearBias

from abc import ABC
import copy
from pathlib import Path
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

REDEFINABLE_FROM_LEARNER_ATTRS = ( 'batch_size', 'loss', 'optimizer', 'lr_sched',
                                   'latent_distribution', )

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
      self.data_config = get_current_configuration( 'data_config' )
      self._ds_mean = torch.FloatTensor( self.data_config.ds_mean ).unsqueeze( dim = 1 ).unsqueeze( dim = 2 )
      self._ds_std = torch.FloatTensor( self.data_config.ds_std ).unsqueeze( dim = 1 ).unsqueeze( dim = 2 )
      self._ds_mean_unsq = self._ds_mean.unsqueeze( dim = 0 )
      self._ds_std_unsq = self._ds_std.unsqueeze( dim = 0 )

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

      # Optimizer:
      # TODO: You will probably have to separate out mapping network (i.e. `self.z_to_w`) and synthesis network
      #       if you want to efficiently and cleanly implement the EWMA of W for the Truncation Trick.
      self._set_optimizer( )

      self.batch_size = self.config.bs_dict[ self.gen_model.curr_res ]

      if self.cond_gen:
        self.labels_one_hot_disc = self._tensor( self.batch_size, self.num_classes )
        self.labels_one_hot_gen = self._tensor( self.batch_size * self.config.gen_bs_mult, self.num_classes )

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

  # TODO: You have a more diverse set of trainable parameters for StyleGAN. Make this work for it.
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