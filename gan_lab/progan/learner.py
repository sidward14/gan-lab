# -*- coding: UTF-8 -*-

"""Learner for ProGANs (Progressively Growing GANs).

  Typical usage example:

  First configure your desired GAN on the command-line:
    go to root directory...
    $ python config.py progan
    $ python data_config.py CelebA-HQ path/to/datasets/celeba_hq

  Then write a custom script (or use train.py):
    from gan_lab import get_current_configuration
    from gan_lab.utils.data_utils import prepare_dataset, prepare_dataloader
    from gan_lab.progan.learner import ProGANLearner

    # get most recent configurations:
    config = get_current_configuration( 'config' )
    data_config = get_current_configuration( 'data_config' )

    # get DataLoader(s)
    train_ds, valid_ds = prepare_dataset( data_config )
    train_dl, valid_dl, z_valid_dl = prepare_dataloader( config, data_config, train_ds, valid_ds )

    # instantiate ProGANLearner and train:
    learner = ProGANLearner( config )
    learner.train( train_dl, valid_dl, z_valid_dl )   # train for config.num_main_iters iterations
    learner.config.num_main_iters = 300000            # this is one example of changing your instantiated learner's configurations
    learner.train( train_dl, valid_dl, z_valid_dl )   # train for another 300000 iterations

Note that the above custom script is just a more flexible alternative to running
train.py (you can, for example, run the above on a Jupyter Notebook). You can
always just run train.py.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from .base import ProGAN
from .architectures import ProGenerator, ProDiscriminator
from _int import get_current_configuration, LearnerConfigCopy
from resnetgan.learner import GANLearner
from utils.latent_utils import gen_rand_latent_vars
from utils.backprop_utils import calc_gp, configure_adam_for_gan
from utils.custom_layers import Conv2dBias, LinearBias

import os
import sys
from abc import ABC
import copy
import logging
import warnings
from pathlib import Path
from functools import partial
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

from tqdm import tqdm
from tqdm.autonotebook import tqdm as tqdma

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

NONREDEFINABLE_ATTRS = ( 'model', 'init_res', 'res_samples', 'res_dataset', 'len_latent',
                         'num_classes', 'class_condition', 'use_auxiliary_classifier',
                         'model_upsample_type', 'model_downsample_type', 'align_corners',
                         'blur_type', 'nonlinearity', 'use_equalized_lr', 'normalize_z',
                         'use_pixelnorm', 'mbstd_group_size', 'use_ewma_gen', )

REDEFINABLE_FROM_LEARNER_ATTRS = ( 'batch_size', 'loss', 'gradient_penalty',
                                   'optimizer', 'lr_sched', 'latent_distribution', )

COMPUTE_EWMA_VIA_HALFLIFE = True
EWMA_SMOOTHING_HALFLIFE = 10.
EWMA_SMOOTHING_BETA = .999

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class ProGANLearner( GANLearner ):
  """GAN Learner specifically designed for ProGAN architectures.

  Once instantiated, the ProGANLearner object's configuration can be changed, but only via
  its self.config attribute (i.e. running 'python config.py [model]' post-instantiation
  will not affect this learner's configuration).
  """
  def __init__( self, config ):
    super( ProGANLearner, self ).__init__( config )

    # Training data's skip-connection resizing:
    self.ds_sc_upsampler = None
    self.ds_sc_downsampler = None
    self.ds_sc_resizer = None

    _model_selected = False
    if self.model == 'ProGAN':

      # If you want to change an attribute in an already-instantiated ProGANLearner's config or data_config,
      # change self.config (below) instead of config and self.data_config (also below) instead of data_config:
      self.config = LearnerConfigCopy( config,
                                       self.__class__.__name__,
                                       NONREDEFINABLE_ATTRS,
                                       REDEFINABLE_FROM_LEARNER_ATTRS )
      # pretrained models loaded later on for evaluation should not require data_config.py to have been run:
      self._is_data_configed = False; self._stats_set = False
      self._update_data_config( raise_exception = False )

      self.latent_distribution = self.config.latent_distribution

      # Instantiate Neural Networks:
      global ProGAN, ProGenerator, ProDiscriminator
      ProGAN = type( 'ProGAN', ( nn.Module, ABC, ), dict( ProGAN.__dict__ ) )
      ProGAN.reset_state( )

      ProGenerator = type( 'ProGenerator', ( ProGAN, ), dict( ProGenerator.__dict__ ) )
      self.gen_model = ProGenerator(
        final_res = self.config.res_samples,
        len_latent = self.config.len_latent,
        upsampler = self.gen_model_upsampler,
        blur_type = self.config.blur_type,
        nl = self.nl,
        num_classes = self.num_classes_gen,
        equalized_lr = self.config.use_equalized_lr,
        normalize_z = self.config.normalize_z,
        use_pixelnorm = self.config.use_pixelnorm
      )

      ProDiscriminator = type( 'ProDiscriminator', ( ProGAN, ), dict( ProDiscriminator.__dict__ ) )
      self.disc_model = ProDiscriminator(
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

      _model_selected = True

    # Training Set-specific Inits:
    self.curr_phase_num = 0

    # Validation Set-specific Inits:
    self.lagged_params = None

    # Whether in progressive growing stage:
    self._progressively_grow = True

    # Print configuration:
    if _model_selected:
      print( '-------- Initialized Model Configuration --------' )
      print( self.config )
      print( '-------------------------------------------------' )
      # print( "  If you would like to alter any of the above configurations,\n" + \
      #        "  please do so via altering your instantiated ProGANLearner().config's attributes." )
      print( '\n    Ready to train!\n' )

  def _apply_lagged_weights( self, m ):
    # TODO: Include support for other learnable layers such as BatchNorm
    _keys = m.state_dict().keys()
    if isinstance( m, ( nn.Linear, nn.Conv2d, LinearBias, Conv2dBias, ) ):
      if 'weight' in _keys:
        m.weight = nn.Parameter( self.lagged_params.values()[ self._param_tensor_num ] )
        self._param_tensor_num += 1
      if 'bias' in _keys:
        m.bias = nn.Parameter( self.lagged_params.values()[ self._param_tensor_num ] )
        self._param_tensor_num += 1

  @torch.no_grad()
  def _update_gen_lagged( self ):
    self.gen_model_lagged = copy.deepcopy( self.gen_model )  # for memory efficiency in GPU
    self.gen_model_lagged.to( self.config.dev )
    self.gen_model_lagged.train()
    if self.beta:
      self.gen_model_lagged.apply( self._apply_lagged_weights )
      # print( f'{self._param_tensor_num} parameters in Generator.' )
      self._param_tensor_num = 0

  # TODO: Implement:
  #         1.) 'ac loss' metric
  #         2.) class determined for random generated sample,
  #         3.) class determined for random real sample (only if `self.ac` is `True`)
  @torch.no_grad()
  def compute_metrics( self, metrics:str, metrics_type:str, z_valid_dl, valid_dl = None ):
    """Metric evaluation, run periodically during training or independently by the user."""

    if not self._is_data_configed:
      self._update_data_config( raise_exception = True )
    self.data_config = get_current_configuration( 'data_config', raise_exception = True )
    _ds_mean_unsq = self.ds_mean.unsqueeze( dim = 0 )
    _ds_std_unsq = self.ds_std.unsqueeze( dim = 0 )

    metrics_type = metrics_type.casefold()
    if metrics_type not in ( 'generator', 'critic', 'discriminator', ):
      raise Exception( 'Invalid metrics_type. Only "generator", "critic", or "discriminator" are accepted.' )
    metrics = [ metric.casefold() for metric in metrics ]

    self.disc_model.to( self.config.metrics_dev )
    self.disc_model.eval()
    self.gen_model.to( self.config.metrics_dev )
    if self.config.use_ewma_gen:
      if metrics_type == 'generator':
        self.gen_model.train()
        self._update_gen_lagged( )
      self.gen_model_lagged.to( self.config.metrics_dev )
      self.gen_model_lagged.eval()
    self.gen_model.eval()

    valid_dataiter = None
    z_valid_dataiter = None
    if valid_dl is not None:
      valid_dataiter = iter( valid_dl )
    if z_valid_dl is not None:
      z_valid_dataiter = iter( z_valid_dl )
      _len_z_valid_dl = len( z_valid_dl )
      _len_z_valid_ds = len( z_valid_dl.dataset )

    if not self.grid_inputs_constructed and 'image grid' in metrics:
      assert ( self.config.img_grid_sz**2 <= _len_z_valid_ds )
      self.rand_idxs = torch.multinomial( input = torch.ones( _len_z_valid_ds, dtype = torch.float32, device = 'cpu' ),
                                          num_samples = self.config.img_grid_sz**2, replacement = False )
    self._img_grid_constructed = False

    metrics_tensors = { metric : torch.zeros( self.batch_size, _len_z_valid_dl, \
                                 device = self.config.metrics_dev, dtype = torch.float32 ) for metric in metrics }

    if z_valid_dl is not None:
      pbarv = tqdma( total = _len_z_valid_ds, unit = ' imgs', dynamic_ncols = True )
      for n in range( _len_z_valid_dl ):
        # Uncomment the below if validation set is taking up too much memory
        # zb = gen_rand_latent_vars( num_samples = self.batch_size, length = LEN_Z, distribution = 'normal', device = self.config.dev )
        zbatch = next( z_valid_dataiter )
        zb = ( zbatch[0] ).to( self.config.metrics_dev )
        gen_labels = None
        if len( zbatch ) > 1: gen_labels = ( zbatch[1] ).to( 'cpu' )
        _len_zb = len( zb )
        _xgenb = self.gen_model( zb )
        if metrics_type == 'generator':
          if 'fake realness' in metrics:
            if self.ac:
              metrics_tensors['fake realness'][:_len_zb, n], _ = self.disc_model( _xgenb )
            else:
              metrics_tensors['fake realness'][:_len_zb, n] = self.disc_model( _xgenb )
          if 'generator loss' in metrics:
            if 'fake realness' in metrics:
              _ygenb = metrics_tensors['fake realness'][:_len_zb, n]
            else:
              if self.ac: _ygenb, _ = self.disc_model( _xgenb )
              else: _ygenb = self.disc_model( _xgenb )
            metrics_tensors['generator loss'][:_len_zb, n] = self.loss_func_gen( _ygenb )
          if 'image grid' in metrics:
            if self.valid_z is None:
              self.valid_z = torch.FloatTensor( self.config.img_grid_sz**2, zb.shape[1] ).to( self.config.metrics_dev )
              if self.valid_label is None and gen_labels is not None and self.config.img_grid_show_labels:
                self.valid_label = torch.LongTensor( self.config.img_grid_sz**2 ).to( 'cpu' )
              _idx = 0

            if not self.grid_inputs_constructed:
              for o in range( _len_zb ):
                if ( n*self.batch_size + o ) in self.rand_idxs:
                  self.valid_z[ _idx ] = zb[ o ]
                  if gen_labels is not None and self.config.img_grid_show_labels:
                    self.valid_label[ _idx ] = gen_labels[ o ]
                  _idx += 1
              if _idx == self.config.img_grid_sz**2:
                self.grid_inputs_constructed = True

            if self.grid_inputs_constructed and not self._img_grid_constructed:
              if self.config.use_ewma_gen:
                # print( 'TIME-AVERAGED GENERATOR OUTPUT:\n------------------------' )
                save_ewma_img_grid_dir = \
                  self.config.save_samples_dir/self.model.casefold().replace( " ", "" )/self.data_config.dataset/'image_grid'/'time_averaged'
                save_ewma_img_grid_dir.mkdir( parents = True, exist_ok = True )
                fig, _ = self.make_image_grid( zs = self.valid_z, labels = self.valid_label, time_average = True,
                                               save_path = str( save_ewma_img_grid_dir/( str( self.gen_metrics_num ) + '.png' ) ) )
                # plt.show( )

                # print( 'ORIGINAL SNAPSHOT GENERATOR OUTPUT:\n--------------------------' )
              save_original_img_grid_dir = \
                self.config.save_samples_dir/self.model.casefold().replace( " ", "" )/self.data_config.dataset/'image_grid'/'original'
              save_original_img_grid_dir.mkdir( parents = True, exist_ok = True )
              fig, _ = self.make_image_grid( zs = self.valid_z, labels = self.valid_label, time_average = False,
                                             save_path = str( save_original_img_grid_dir/( str( self.gen_metrics_num ) + '.png' ) ) )
              # plt.show( )

              self._img_grid_constructed = True
          self.gen_metrics_num += 1 if n == ( _len_z_valid_dl - 1 ) else 0
        if metrics_type in ( 'critic', 'discriminator', ):
          if 'fake realness' in metrics:
            if self.ac:
              metrics_tensors['fake realness'][:_len_zb, n], _ = self.disc_model( _xgenb )
            else:
              metrics_tensors['fake realness'][:_len_zb, n] = self.disc_model( _xgenb )
          if valid_dl is not None:
            xbatch = next( valid_dataiter )
            # xb = ( xbatch[0] ).to( self.config.metrics_dev )
            xb = xbatch[0]
            # Fade in the real images the same way the generated images are being faded in:
            if self.gen_model.fade_in_phase:
              if self.config.bit_exact_resampling:
                xb_low_res = xb.clone().mul( _ds_std_unsq ).add( _ds_mean_unsq )
                for sample_idx in range( len( xb_low_res ) ):
                  xb_low_res[ sample_idx ] = self.ds_sc_resizer( xb_low_res[ sample_idx ] )
                xb = torch.add( xb_low_res.mul( 1. - self.gen_model.alpha ), xb.mul( self.gen_model.alpha ) )
              else:
                xb = self.ds_sc_upsampler( self.ds_sc_downsampler( xb ) ) * ( 1. - self.gen_model.alpha ) + \
                                           xb * ( self.gen_model.alpha )
            xb = xb.to( self.config.metrics_dev )
            if self.ac: real_labels = ( xbatch[1] ).to( 'cpu' )
            if 'real realness' in metrics:
              if self.ac:
                metrics_tensors['real realness'][:_len_zb, n], _ = self.disc_model( xb )
              else:
                metrics_tensors['real realness'][:_len_zb, n] = self.disc_model( xb )
            if 'discriminator loss' in metrics:
              if 'fake realness' in metrics:
                _ygenb = metrics_tensors['fake realness'][:_len_zb, n]
              else:
                if self.ac: _ygenb, _ = self.disc_model( _xgenb )
                else: _ygenb = self.disc_model( _xgenb )
              if 'real realness' in metrics:
                _yb = metrics_tensors['real realness'][:_len_zb, n]
              else:
                if self.ac: _yb, _ = self.disc_model( xb )
                else: _yb = self.disc_model( xb )
              metrics_tensors['discriminator loss'][:_len_zb, n] = \
                self.loss_func_disc(
                  _ygenb,
                  _yb
                )
              # TODO: Include gradient penalty despite the fact that this method is under @torch.no_grad().
              # if self.gradient_penalty is not None:
              #   metrics_tensors['generator loss'][:len(zb), n] += self.gp_func( _xgenb, xb )
          self.disc_metrics_num += 1 if n == ( _len_z_valid_dl - 1 ) else 0
        pbarv.set_description( ' Img' )
        pbarv.update( _len_zb )
    pbarv.close()
    metrics_vals = [
      ( vals_tensor.sum() / _len_z_valid_ds ).item() \
      for vals_tensor in metrics_tensors.values()
    ]
    _max_len = '%-' + str( max( [ len( s ) for s in metrics ] ) + 3 ) + 's'
    metrics_vals = [ '    ' + ( _max_len % ( metric + ':' ) ) + '%.4g' % metrics_vals[n] + '\n' for n, metric in enumerate( metrics ) if metric != 'image grid' ]

    self.gen_model.to( self.config.dev )
    self.disc_model.to( self.config.dev )

    self.gen_model.train()
    self.disc_model.train()

    return metrics_vals

  def train( self, train_dl, valid_dl = None, z_valid_dl = None,
             num_main_iters = None, num_gen_iters = None, num_disc_iters = None ):

    """Efficient & fast implementation of ProGAN training (at the expense of some messy/repetitive code).

    Arguments num_main_iters, num_gen_iters, and num_disc_iters are taken from self.config if not
    explicitly specified when running this method (this is usually the case). Typically, one just specifies
    train_dl (and maybe valid_dl and z_valid_dl as well if one wants to periodically evaluate metrics).
    """

    # If custom number of iterations are not input, use self.config's number of iterations as the default
    if num_main_iters is None:
      num_main_iters = self.config.num_main_iters
    if num_gen_iters is None:
      num_gen_iters = self.config.num_gen_iters
    if num_disc_iters is None:
      num_disc_iters = self.config.num_disc_iters

    self.num_main_iters = num_main_iters
    self.dataset_sz = len( train_dl.dataset )

    self._update_data_config( raise_exception = True )
    _ds_mean_unsq = self.ds_mean.unsqueeze( dim = 0 )
    _ds_std_unsq = self.ds_std.unsqueeze( dim = 0 )

    self.gen_model.to( self.config.dev )
    self.gen_model.train()
    self.disc_model.to( self.config.dev )
    self.disc_model.train()
    if self.config.use_ewma_gen:
      self.gen_model_lagged.to( self.config.metrics_dev )
      self.gen_model_lagged.train()

    # Initialize number of images before transition to next fade-in/stabilization phase:
    if self.not_trained_yet or self.pretrained_model:
      if ( self.config.nimg_transition % self.batch_size ) != 0:
        self.nimg_transition = self.batch_size * ( int( self.config.nimg_transition / self.batch_size ) + 1 )
      else:
        self.nimg_transition = self.config.nimg_transition

    if self.not_trained_yet:
      self.nimg_transition_lst = [ self.nimg_transition ]

    # Initialize validation EWMA smoothing of generator weights & biases:
    if self.not_trained_yet or self.pretrained_model:
      self.beta = None
      if self.config.use_ewma_gen:
        if COMPUTE_EWMA_VIA_HALFLIFE:
          self.beta = self.get_smoothing_ewma_beta( half_life = EWMA_SMOOTHING_HALFLIFE )
        else:
          self.beta = EWMA_SMOOTHING_BETA
        # TODO: Can this be done more memory-efficiently (w/o sacrificing speed)?
        with torch.no_grad():
          # dict is not that much slower than list (according to `timeit` tests)
          self.lagged_params = IndexedOrderedDict( self.gen_model.named_parameters( ) )

    # Set scheduler on every run:
    if self.sched_bool:
      if not self.pretrained_model:
        self.sched_stop_step = 0
      self._set_scheduler( )
    elif self.pretrained_model:
      self.scheduler_gen = self._scheduler_gen_state_dict  #  = None
      self.scheduler_disc = self._scheduler_disc_state_dict  #  = None

    # Allows one to start from where they left off:
    if self.not_trained_yet:
      print( 'STARTING FROM ITERATION 0:\n' )
      self.train_dataiter = iter( train_dl )
    else:
      print( 'CONTINUING FROM WHERE YOU LEFT OFF:\n' )
      if self.pretrained_model:
        train_dl.batch_sampler.batch_size = self.batch_size
        train_dl.dataset.transforms.transform.transforms = \
          self.increase_real_data_res( transforms_lst = train_dl.dataset.transforms.transform.transforms )

        self.train_dataiter = iter( train_dl )

        if valid_dl is not None:
          valid_dl.batch_sampler.batch_size = self.batch_size
          valid_dl.dataset.transforms.transform.transforms = \
            self.increase_real_data_res( transforms_lst = valid_dl.dataset.transforms.transform.transforms )

        if z_valid_dl is not None:
          z_valid_dl.batch_sampler.batch_size = self.batch_size

    if self.pretrained_model and self.gen_model.fade_in_phase:
      if self.config.bit_exact_resampling:
        for transform in self.train_dataiter._dataset.transforms.transform.transforms:
          if isinstance( transform, transforms.Normalize ):
            _nrmlz_transform = transform
        self.ds_sc_resizer = \
          self.get_real_data_skip_connection_transforms( self.data_config.dataset_downsample_type, _nrmlz_transform )
      else:
        # matches the model's skip-connection upsampler
        self.ds_sc_upsampler = lambda xb: F.interpolate( xb, scale_factor = 2, mode = 'nearest' )
        # matches the dataset's downsampler
        if self.data_config.dataset_downsample_type in ( Image.BOX, Image.BILINEAR, ):
          self.ds_sc_downsampler = lambda xb: F.avg_pool2d( xb, kernel_size = 2, stride = 2 )
        elif self.data_config.dataset_downsample_type == Image.NEAREST:
          self.ds_sc_downsampler = lambda xb: F.interpolate( xb, scale_factor = .5, mode = 'nearest' )

      self.delta_alpha = self.batch_size / ( ( self.nimg_transition / num_disc_iters ) - self.batch_size )

    if self.tot_num_epochs is None:
      _tmpd1 = self.nimg_transition // self.config.bs_dict[ self.config.init_res ]
      _tmpd20 = { k:v for k,v in self.config.bs_dict.items() if self.config.init_res < k < self.config.res_samples }
      _tmpd20 = np.unique( np.array( list( _tmpd20.values() ) ), return_counts = True )
      _tmpd2 = ( self.nimg_transition // _tmpd20[0] ) * 2 * _tmpd20[1]
      _tmpd3 = num_main_iters - ( _tmpd1 + _tmpd2.sum() )
      self.tot_num_epochs = _tmpd1*self.config.bs_dict[ self.config.init_res ] + ( _tmpd2*_tmpd20[0] ).sum() + _tmpd3*self.config.bs_dict[ self.config.res_samples ]
      self.tot_num_epochs *= num_disc_iters
      self.tot_num_epochs //= self.dataset_sz // self.batch_size * self.batch_size
      self.tot_num_epochs += 1

    pbar = tqdm( total = self.dataset_sz // self.batch_size * self.batch_size, unit = ' imgs' )

    # ------------------------------------------------------------------------ #

    try:

      for itr in range( num_main_iters ):

        if self.sched_bool:
          with warnings.catch_warnings():
            warnings.simplefilter( 'ignore' )
            tqdm_lr = '%9s' % ( '%g' % self.scheduler_disc.get_lr()[0] )
            tqdm_lr += '%9s' % ( '%g' % self.scheduler_gen.get_lr()[0] )
            # tqdm_desc += f'Generator LR: { " ".join( [ str(s) for s in self.scheduler_gen.get_lr() ] ) } | ' + \
            #              f'Discriminator LR: { " ".join( [ str(s) for s in self.scheduler_disc.get_lr() ] ) }'
            # print( f'Generator LR: {*self.scheduler_gen.get_lr()} |', \
            #        f'Discriminator LR: {*self.scheduler_disc.get_lr()}'
            # )
        else:
          tqdm_lr = '%9s' % ( '%g' % self.config.lr_base )
          tqdm_lr += '%9s' % ( '%g' % self.config.lr_base )

        # these are set to `False` for the Generator because you don't need
        # the Discriminator's parameters' gradients when chain-ruling back to the generator
        for p in self.disc_model.parameters(): p.requires_grad_( True )

        # Determine whether it is time to switch to next fade-in/stabilization phase:
        if self.gen_model.curr_res < self.gen_model.final_res:
          if self.curr_img_num == sum( self.nimg_transition_lst ):
            self.curr_phase_num += 1
            if self.curr_phase_num % 2 == 1:
              _prev_res = self.gen_model.curr_res

              self.gen_model.zero_grad()
              self.disc_model.zero_grad()

              self.gen_model.increase_scale()
              self.disc_model.increase_scale()

              # Generator and Discriminator state data must match:
              assert self.gen_model.cls_base.__dict__ == \
                    self.disc_model.cls_base.__dict__

              self.gen_model.to( self.config.dev )
              self.disc_model.to( self.config.dev )

              # ---------------- #

              # Update Optimizer:
              self._set_optimizer( )

              # Update Scheduler:
              if self.sched_bool:
                self.sched_stop_step += self.scheduler_gen._step_count
                self._set_scheduler( )

              # ---------------- #

              print( f'\n\n\nRESOLUTION INCREASED FROM {_prev_res}x{_prev_res} to ' + \
                    f'{int( self.gen_model.curr_res )}x{int( self.gen_model.curr_res )}\n' )
              print( f'FADING IN {int( self.gen_model.curr_res )}x' + \
                    f'{int( self.gen_model.curr_res )} RESOLUTION...\n' )
              print( ('\n' + '%9s' * 8 ) % ( 'Epoch', 'Res', 'Phase', 'D LR', 'G LR', 'D Loss', 'G Loss', 'Itr' ) )

              # ---------------- #

              # Update resolution-specific batch size:
              self.batch_size = self.config.bs_dict[ self.gen_model.curr_res ]

              # ---------------- #

              self._set_loss( )

              # ---------------- #

              train_dl.batch_sampler.batch_size = self.batch_size
              # self.train_dataiter._index_sampler.batch_size = self.batch_size

              train_dl.dataset.transforms.transform.transforms = \
                self.increase_real_data_res( transforms_lst = train_dl.dataset.transforms.transform.transforms )
              if isinstance( self.train_dataiter, torch.utils.data.dataloader._MultiProcessingDataLoaderIter ):
                self.train_dataiter = iter( train_dl )  # TODO: this makes you re-use your data before epoch end.

              if self.config.bit_exact_resampling:
                for transform in self.train_dataiter._dataset.transforms.transform.transforms:
                  if isinstance( transform, transforms.Normalize ):
                    _nrmlz_transform = transform
                self.ds_sc_resizer = \
                  self.get_real_data_skip_connection_transforms( self.data_config.dataset_downsample_type, _nrmlz_transform )
              else:
                # matches the model's skip-connection upsampler
                self.ds_sc_upsampler = lambda xb: F.interpolate( xb, scale_factor = 2, mode = 'nearest' )
                # matches the dataset's downsampler
                if self.data_config.dataset_downsample_type in ( Image.BOX, Image.BILINEAR, ):
                  self.ds_sc_downsampler = lambda xb: F.avg_pool2d( xb, kernel_size = 2, stride = 2 )
                elif self.data_config.dataset_downsample_type == Image.NEAREST:
                  self.ds_sc_downsampler = lambda xb: F.interpolate( xb, scale_factor = .5, mode = 'nearest' )

              # ---------------- #

              if valid_dl is not None:
                valid_dl.batch_sampler.batch_size = self.batch_size
                valid_dl.dataset.transforms.transform.transforms = \
                  self.increase_real_data_res( transforms_lst = valid_dl.dataset.transforms.transform.transforms )

              # ---------------- #

              if z_valid_dl is not None:
                z_valid_dl.batch_sampler.batch_size = self.batch_size

              # ---------------- #

              # Number of images before switching to next fade-in/stabilization phase:
              if ( self.config.nimg_transition % self.batch_size ) != 0:
                self.nimg_transition = self.batch_size * ( int( self.config.nimg_transition / self.batch_size ) + 1 )
              else:
                self.nimg_transition = self.config.nimg_transition

              # ---------------- #

              self.delta_alpha = self.batch_size / ( ( self.nimg_transition / num_disc_iters ) - self.batch_size )
              self.gen_model.alpha = 0  # this applies to both networks simultaneously

              # ---------------- #

              if self.config.use_ewma_gen:
                if COMPUTE_EWMA_VIA_HALFLIFE:
                  self.beta = self.get_smoothing_ewma_beta( half_life = EWMA_SMOOTHING_HALFLIFE )

                # Update `self.lagged_params`:
                with torch.no_grad():
                  self.lagged_params[ 'prev_torgb.conv2d.weight' ] = self.lagged_params.pop( 'torgb.conv2d.weight' )
                  self.lagged_params[ 'prev_torgb.conv2d.bias' ] = self.lagged_params.pop( 'torgb.conv2d.bias' )
                  for name, param in IndexedOrderedDict( self.gen_model.named_parameters() ).items():
                    if name not in self.lagged_params:
                      self.lagged_params[ name ] = param
                  # Order the names to match that of `self.gen_model`'s order:
                  for name, param in IndexedOrderedDict( self.gen_model.named_parameters() ).items():
                    if 'fc_mapping_model' in name:
                      self.lagged_params.move_to_end( name, last = True )
                  for name, param in IndexedOrderedDict( self.gen_model.named_parameters() ).items():
                    if name == 'torgb.conv2d.weight':
                      self.lagged_params.move_to_end( name, last = True )
                  for name, param in IndexedOrderedDict( self.gen_model.named_parameters() ).items():
                    if name == 'torgb.conv2d.bias':
                      self.lagged_params.move_to_end( name, last = True )
                  for name, param in IndexedOrderedDict( self.gen_model.named_parameters() ).items():
                    if name == 'prev_torgb.conv2d.weight':
                      self.lagged_params.move_to_end( name, last = True )
                  for name, param in IndexedOrderedDict( self.gen_model.named_parameters() ).items():
                    if name == 'prev_torgb.conv2d.bias':
                      self.lagged_params.move_to_end( name, last = True )

            else:
              # Update Optimizer:
              self._set_optimizer( )

              # Update Scheduler:
              if self.sched_bool:
                self.sched_stop_step += self.scheduler_gen._step_count
                self._set_scheduler( )

              # ---------------- #

              print( '\nSTABILIZING...\n' )

            self.nimg_transition_lst.append( self.nimg_transition )

          else:
            if not itr:
              if self.gen_model.fade_in_phase:
                print( f'\nFADING IN {int( self.gen_model.curr_res )}x' + \
                      f'{int( self.gen_model.curr_res )} RESOLUTION...\n' )
              else:
                print( '\nSTABILIZING...\n' )

        if not itr and not self.progressively_grow:
          print( '\nSTABILIZING (FINAL)...\n' )

        # Final phase:
        if self.curr_img_num == sum( self.nimg_transition_lst ):
          # Update Optimizer:
          self._set_optimizer( )

          # Update Scheduler:
          if self.sched_bool:
            self.sched_stop_step += self.scheduler_gen._step_count
            self._set_scheduler( )

          # ---------------- #

          print( '\nSTABILIZING (FINAL)...\n' )

          self.curr_phase_num += 1
          self.nimg_transition_lst.append( np.inf )

          self._progressively_grow = False

        #------------------------- TRAIN DISCRIMINATOR ----------------------------

        # loss_train_disc = None
        for disc_iter in range( num_disc_iters ):
          self.disc_model.zero_grad()

          # Sample latent vector z:
          if self.cond_gen:
            # Class-conditioning in the latent space:
            # TODO: Implement embedding-style conditioning from "Which Training Methods for
            #       GANs do actually Converge" & discriminator conditioning.
            gen_labels = torch.randint( 0, self.num_classes, ( self.batch_size, 1, ), dtype = torch.int64, device = self.config.dev )
            zb = gen_rand_latent_vars( num_samples = self.batch_size, length = self.config.len_latent,
                                      distribution = self.latent_distribution, device = self.config.dev )
            self.labels_one_hot_disc.zero_()
            self.labels_one_hot_disc.scatter_( 1, gen_labels, 1 )
            if self.ac: gen_labels.squeeze_()
            zb = torch.cat( ( zb, self.labels_one_hot_disc, ), dim = 1 )
          else:
            zb = gen_rand_latent_vars( num_samples = self.batch_size, length = self.config.len_latent,
                                      distribution = self.latent_distribution, device = self.config.dev )
          with torch.no_grad(): zbv = zb  # makes sure to totally freeze the generator when training discriminator
          _xgenb = self.gen_model( zbv ).detach()

          # Sample real data x:
          batch = next( self.train_dataiter, None )
          if batch is None:
            self.curr_epoch_num += 1
            pbar.close()
            print( f'\n\nEPOCH # {self.curr_epoch_num - 1} COMPLETE. BEGIN EPOCH #', \
                  f'{self.curr_epoch_num}\n' )
            print( ('\n' + '%9s' * 8 ) % ( 'Epoch', 'Res', 'Phase', 'D LR', 'G LR', 'D Loss', 'G Loss', 'Itr' ) )
            pbar = tqdm( total = self.dataset_sz // self.batch_size * self.batch_size, unit = ' imgs' )
            self.train_dataiter = iter( train_dl )
            batch = next( self.train_dataiter )
          # xb = ( batch[0] ).to( self.config.dev )
          xb = batch[0]

          # Fade in the real images the same way the generated images are being faded in:
          if self.gen_model.fade_in_phase:
            with torch.no_grad():
              if self.config.bit_exact_resampling:
                xb_low_res = xb.clone().mul( _ds_std_unsq ).add( _ds_mean_unsq )
                for sample_idx in range( len( xb_low_res ) ):
                  xb_low_res[ sample_idx ] = self.ds_sc_resizer( xb_low_res[ sample_idx ] )
                xb = torch.add( xb_low_res.mul( 1. - self.gen_model.alpha ), xb.mul( self.gen_model.alpha ) )
              else:
                xb = self.ds_sc_upsampler( self.ds_sc_downsampler( xb ) ) * ( 1. - self.gen_model.alpha ) + \
                                          xb * ( self.gen_model.alpha )
          xb = xb.to( self.config.dev )
          if self.ac: real_labels = ( batch[1] ).to( self.config.dev )

          # Forward prop:
          if self.ac:
            discriminative_gen, gen_preds = self.disc_model( _xgenb )
            discriminative_real, real_preds = self.disc_model( xb )
          else:
            discriminative_gen = self.disc_model( _xgenb )
            discriminative_real = self.disc_model( xb )

          if self.loss == 'wgan':
            loss_train_disc = ( discriminative_gen - discriminative_real ).mean()
          elif self.loss in ( 'nonsaturating', 'minimax' ):
            loss_train_disc = \
              F.binary_cross_entropy_with_logits( input = discriminative_gen,
                                                  target = self._dummy_target_gen,
                                                  reduction = 'mean' ) + \
              F.binary_cross_entropy_with_logits( input = discriminative_real,
                                                  target = self._dummy_target_real,
                                                  reduction = 'mean' )

          if self.ac:
            loss_train_disc += \
              ( self.loss_func_aux( gen_preds, gen_labels ) + \
                self.loss_func_aux( real_preds, real_labels )
              ).mean() * self.config.ac_disc_scale

          if self.gradient_penalty is not None:
            loss_train_disc += self.calc_gp( _xgenb, xb )

          if self.eps:
            loss_train_disc += ( discriminative_real**2 ).mean() * self.config.eps_drift

          # Backprop:
          loss_train_disc.backward()  # compute the gradients
          self.opt_disc.step()        # update the parameters you specified to the optimizer with backprop
          # self.opt_disc.zero_grad()

          # Compute metrics for discriminator (validation metrics should be for entire validation set):
          metrics_vals = []
          _valid_title = []
          if z_valid_dl is not None and valid_dl is not None and self.config.disc_metrics:
            if ( disc_iter == num_disc_iters - 1 ) and ( ( itr + 1 ) % self.config.num_iters_valid == 0 or itr == 0 ):
              if itr != 0:
                end = timer(); print( f'\n\nTime since last Validation Set: {end - start} seconds.' )

              metrics_vals = self.compute_metrics(
                metrics = self.config.disc_metrics, metrics_type = 'Discriminator',
                z_valid_dl = z_valid_dl, valid_dl = valid_dl
              )
              _valid_title = [ '|\n', 'Discriminator Validation Metrics:\n' ]
              print( *_valid_title, *metrics_vals )

          tqdm_loss_disc = '%9.4g' % loss_train_disc.item()
          if itr:
            tqdm_desc = '%9s' % f'{self.curr_epoch_num}/{self.tot_num_epochs}'
            tqdm_desc += '%9s' % f'{self.gen_model.curr_res}X{self.gen_model.curr_res}'
            tqdm_desc += '%9s' % ( 'Fade In' if self.gen_model.fade_in_phase else 'Stab.' )
            tqdm_desc += tqdm_lr
            tqdm_desc += tqdm_loss_disc + tqdm_loss_gen
            tqdm_desc += '%9s' % itr
            tqdm_desc += '    Img'
            pbar.set_description( tqdm_desc )

          pbar.update( xb.shape[0] )

          self.curr_dataset_batch_num += 1
          self.curr_img_num += self.batch_size

        #------------------------ TRAIN GENERATOR --------------------------

        # these are set to `False` for the Generator because you don't need
        # the Discriminator's parameters' gradients when chain-ruling back to the generator
        for p in self.disc_model.parameters(): p.requires_grad_( False )

        # loss_train_gen = None
        for gen_iter in range( num_gen_iters ):
          self.gen_model.zero_grad()

          # Sample latent vector z:
          if self.cond_gen:
            # Class-conditioning in the latent space:
            # TODO: Implement embedding-style conditioning from "Which Training Methods for
            #       GANs do actually Converge" & discriminator conditioning.
            gen_labels = torch.randint( 0, self.num_classes, ( self.batch_size * self.config.gen_bs_mult, 1, ), dtype = torch.int64, device = self.config.dev )
            zb = gen_rand_latent_vars( num_samples = self.batch_size * self.config.gen_bs_mult, length = self.config.len_latent,
                                      distribution = self.latent_distribution, device = self.config.dev )
            self.labels_one_hot_gen.zero_()
            self.labels_one_hot_gen.scatter_( 1, gen_labels, 1 )
            if self.ac: gen_labels.squeeze_()
            zb = torch.cat( ( zb, self.labels_one_hot_gen, ), dim = 1 )
          else:
            zb = gen_rand_latent_vars( num_samples = self.batch_size * self.config.gen_bs_mult, length = self.config.len_latent,
                                      distribution = self.latent_distribution, device = self.config.dev )
          zb.requires_grad_( True )

          # Forward prop:
          if self.ac:
            loss_train_gen, gen_preds = self.disc_model( self.gen_model( zb ) )
          else:
            loss_train_gen = self.disc_model( self.gen_model( zb ) )

          if self.loss == 'wgan':
            loss_train_gen = -loss_train_gen.mean()
          elif self.loss == 'nonsaturating':
            loss_train_gen = F.binary_cross_entropy_with_logits(
              input = loss_train_gen,
              target = self._dummy_target_real,
              reduction = 'mean'
            )
          elif self.loss == 'minimax':
            loss_train_gen = -F.binary_cross_entropy_with_logits(
              input = loss_train_gen,
              target = self._dummy_target_gen,
              reduction = 'mean'
            )

          if self.ac:
            loss_train_gen += self.loss_func_aux( gen_preds, gen_labels ).mean() * self.config.ac_gen_scale

          # Backprop:
          loss_train_gen.backward()  # compute the gradients
          # loss_train_gen = -loss_train_gen
          self.opt_gen.step()        # update the parameters you specified to the optimizer with backprop
          # self.opt_gen.zero_grad()

          # Calculate validation EWMA smoothing of generator weights & biases:
          # TODO: Should this not be applied to the biases (i.e. just the weights)?
          if self.config.use_ewma_gen:
            with torch.no_grad():
              for name, param in self.gen_model.named_parameters():
                if self.beta:
                  self.lagged_params[ name ] = param * ( 1. - self.beta ) + \
                                              self.lagged_params[ name ] * ( self.beta )
                else:
                  self.lagged_params[ name ] = param

          # Compute metrics for generator (validation metrics should be for entire validation set):
          metrics_vals = []
          _valid_title = []
          if z_valid_dl is not None and self.config.gen_metrics:
            if ( gen_iter == num_gen_iters - 1 ) and ( ( itr + 1 ) % self.config.num_iters_valid == 0 or itr == 0 ):
              metrics_vals = self.compute_metrics(
                metrics = self.config.gen_metrics, metrics_type = 'Generator',
                z_valid_dl = z_valid_dl, valid_dl = None
              )
              _valid_title = [ '|\n', 'Generator Validation Metrics:\n' ]
              print( *_valid_title, *metrics_vals )
              if itr:
                print( ('\n' + '%9s' * 8 ) % ( 'Epoch', 'Res', 'Phase', 'D LR', 'G LR', 'D Loss', 'G Loss', 'Itr' ) )

              start = timer()

          tqdm_loss_gen = '%9.4g' % loss_train_gen.item()
          if itr:
            tqdm_desc = '%9s' % f'{self.curr_epoch_num}/{self.tot_num_epochs}'
            tqdm_desc += '%9s' % f'{self.gen_model.curr_res}X{self.gen_model.curr_res}'
            tqdm_desc += '%9s' % ( 'Fade In' if self.gen_model.fade_in_phase else 'Stab.' )
            tqdm_desc += tqdm_lr
            tqdm_desc += tqdm_loss_disc + tqdm_loss_gen
            tqdm_desc += '%9s' % itr
            tqdm_desc += '    Img'
            pbar.set_description( tqdm_desc )

        if not itr:
          print( ('\n' + '%9s' * 8 ) % ( 'Epoch', 'Res', 'Phase', 'D LR', 'G LR', 'D Loss', 'G Loss', 'Itr' ) )
        # pbar.set_postfix( tqdm_desc )
        # tqdm.write( tqdm_desc )
        
        # Update fading-in paramater alpha:
        if self.gen_model.fade_in_phase:
          self.gen_model.alpha += self.delta_alpha  # this affects both networks

        if self.sched_bool:
          self.scheduler_gen.step()
          self.scheduler_disc.step()

        if self.not_trained_yet:
          self.not_trained_yet = False

        # Save model every self.config.num_iters_save_model iterations:
        if ( itr + 1 ) % self.config.num_iters_save_model == 0:
          self._set_optimizer( )  # for niche case when training ends right when alpha becomes 1

          # update time-averaged generator
          with torch.no_grad():
            self.gen_model.to( 'cpu' )
            if self.config.use_ewma_gen:
              self._update_gen_lagged( )
              self.gen_model_lagged.to( 'cpu' )
              self.gen_model_lagged.eval()

          self.gen_model.eval()
          self.disc_model.to( 'cpu' )
          self.disc_model.eval()

          self.save_model( self.config.save_model_dir/( self.model.casefold().replace( " ", "" ) + '_model.tar' ) )

          self.gen_model.to( self.config.dev )
          self.gen_model.train()
          self.disc_model.to( self.config.dev )
          self.disc_model.train()
          if self.config.use_ewma_gen:
            self.gen_model_lagged.to( self.config.metrics_dev )
            self.gen_model_lagged.train()


    except KeyboardInterrupt:
      
      pbar.close()

      self._set_optimizer( )  # for niche case when training ends right when alpha becomes 1

      # update time-averaged generator
      with torch.no_grad():
        self.gen_model.to( 'cpu' )
        if self.config.use_ewma_gen:
          self._update_gen_lagged( )
          self.gen_model_lagged.to( 'cpu' )
          self.gen_model_lagged.eval()

      self.gen_model.eval()
      self.disc_model.to( 'cpu' )
      self.disc_model.eval()

      self.save_model( self.config.save_model_dir/( self.model.casefold().replace( " ", "" ) + '_model.tar' ) )

      print( f'\nTraining interrupted. Saved latest checkpoint into "{self.config.save_model_dir}/".\n' )

      try:
        sys.exit( 0 )
      except SystemExit:
        os._exit( 0 )


    pbar.close()

    self._set_optimizer( )  # for niche case when training ends right when alpha becomes 1

    # update time-averaged generator
    with torch.no_grad():
      self.gen_model.to( 'cpu' )
      if self.config.use_ewma_gen:
        self._update_gen_lagged( )
        self.gen_model_lagged.to( 'cpu' )
        self.gen_model_lagged.eval()

    self.gen_model.eval()
    self.disc_model.to( 'cpu' )
    self.disc_model.eval()

  # .......................................................................... #

  def _set_scheduler( self ):
    if self._lr_sched == 'resolution dependent':
      self.scheduler_fn = lambda _: self.config.lr_fctr_dict[ self.gen_model.curr_res ]
    # self.tot_num_epochs = ( self.batch_size * num_main_iters * num_disc_iters * 1. ) / self.dataset_sz
    elif self._lr_sched == 'linear decay':
      # self.scheduler_fn = lambda epoch: 1. - ( epoch + self.sched_stop_step ) * ( 1. / ( self.tot_num_epochs//1 ) )
      self.scheduler_fn = lambda main_iter: 1. - ( main_iter + self.sched_stop_step ) * ( 1. / self.num_main_iters )
    elif self._lr_sched == 'custom':
      self.scheduler_fn = eval( self.config.lr_sched_custom )
    # TODO: add more types of LR scheduling
    else:
      raise ValueError( "config does not support this LR scheduler.\n" + \
                        "Currently supported LR Schedulers are: [ 'resolution dependent', 'linear decay', 'custom' ]" )

    self.scheduler_gen = \
      torch.optim.lr_scheduler.LambdaLR( self.opt_gen, self.scheduler_fn, last_epoch = -1 )
    self.scheduler_disc = \
      torch.optim.lr_scheduler.LambdaLR( self.opt_disc, self.scheduler_fn, last_epoch = -1 )

    with warnings.catch_warnings():
      warnings.simplefilter( 'ignore', UserWarning )
      self._scheduler_gen_state_dict = self.scheduler_gen.state_dict()
      self._scheduler_disc_state_dict = self.scheduler_disc.state_dict()
      if self.pretrained_model and not self._sched_state_dict_set:
        self.scheduler_gen.load_state_dict( self._scheduler_gen_state_dict )
        self.scheduler_disc.load_state_dict( self._scheduler_disc_state_dict )
        self._scheduler_gen_state_dict = self.scheduler_gen.state_dict()
        self._scheduler_disc_state_dict = self.scheduler_disc.state_dict()
        self._sched_state_dict_set = True

  def _set_optimizer( self ):
    if self._optimizer == 'adam':
      adam_gan = configure_adam_for_gan(
        lr_base = self.config.lr_base,
        betas = ( self.config.beta1, self.config.beta2 ),
        eps = self.config.eps,
        wd = self.config.wd
      )
      if self.gen_model.fade_in_phase:
        self.opt_gen = adam_gan( params = self.gen_model.parameters() )
        self.opt_disc = adam_gan( params = self.disc_model.parameters() )
      else:
        # don't need `prev_torgb` and `prev_fromrgb` during stabilization
        self.opt_gen = adam_gan(
          params = self.gen_model.most_parameters(
            excluded_params = [ 'prev_torgb.conv2d.weight', 'prev_torgb.conv2d.bias' ]
          )
        )
        self.opt_disc = adam_gan(
          params = self.disc_model.most_parameters(
            excluded_params = [ 'prev_fromrgb.0.conv2d.weight', 'prev_fromrgb.0.conv2d.bias' ]
          )
        )
    elif self._optimizer == 'rmsprop':
      raise NotImplementedError( 'RMSprop optimizer not yet implemented.' )  # TODO:
    elif self._optimizer == 'momentum':
      raise NotImplementedError( 'Momentum optimizer not yet implemented.' )  # TODO:
    elif self._optimizer == 'sgd':
      raise NotImplementedError( 'SGD optimizer not yet implemented.' )  # TODO:
    else:
      raise ValueError( "config does not support this optimizer.\n" + \
                        "Supported Optimizers are: [ 'adam', 'rmsprop', 'momentum', 'sgd' ]" )

  # .......................................................................... #

  def increase_real_data_res( self, transforms_lst:list ):
    if not self._is_data_configed:
      self._update_data_config( raise_exception = True )

    _num_rsz = 0
    for n, transform in enumerate( transforms_lst ):
      if isinstance( transform, transforms.Resize ):
        _num_rsz += 1
        if _num_rsz < 2:
          transforms_lst[n] = transforms.Resize( size = ( self.gen_model.curr_res, self.gen_model.curr_res, ),
                                                 interpolation = self.data_config.dataset_downsample_type )
        else:
          raise RuntimeWarning( 'Warning: More than 1 `Resize` transform found; only resized the first `Resize` in transforms list.' )
    return transforms_lst

  def get_real_data_skip_connection_transforms(self, ds_downsampler_type, nrmlz_transform ):
    return transforms.Compose( [ transforms.ToPILImage(),
                                 transforms.Resize( size = ( self.disc_model.prev_res, self.disc_model.prev_res, ),
                                                    interpolation = ds_downsampler_type ),
                                 transforms.Resize( size = ( self.disc_model.curr_res, self.disc_model.curr_res, ),
                                                             interpolation = Image.NEAREST ),
                                 transforms.ToTensor(),
                                 nrmlz_transform
                               ] )

  def get_smoothing_ewma_beta( self, half_life ):
    assert isinstance( half_life, float )

    return .5 ** ( ( self.batch_size * self.config.gen_bs_mult ) / ( half_life * 1000. ) ) if half_life > 0. else 0.

  # .......................................................................... #

  # def reset_progan_state( self ):
  #   self.gen_model.cls_base.reset_state( )  # this applies to both networks simultaneously

  @property
  def progressively_grow( self ):
    return self._progressively_grow

  @progressively_grow.setter
  def progressively_grow( self, new_progressively_grow ):
    assert isinstance( new_progressively_grow, bool )
    self._progressively_grow = new_progressively_grow
    # TODO:
    raise NotImplementedError( 'Setter self.progressively_grow not yet fully implemented.' )

  # .......................................................................... #

  @torch.no_grad()
  def plot_sample( self, z_test, label = None, time_average = True ):
    """Plots and shows 1 sample from input latent code."""
    if self.ds_mean is None or self.ds_std is None:
      raise ValueError( "This model does not hold any information about your dataset's mean and/or std.\n" + \
                        "Please provide these (either from your current data configuration or from your pretrained model)." )

    if z_test.dim() == 2:
      if z_test.shape[0] != 1:
        raise IndexError( 'This method only permits plotting 1 generated sample at a time.' )
    elif z_test.dim() != 1:
      raise IndexError( 'Incorrect dimensions of input latent vector. Must be either `dim == 1` or `dim == 2`.' )
    if not self.cond_gen:
      if z_test.shape[-1] != self.config.len_latent:
        message = f"Input latent vector must be of size {self.config.len_latent}."
        raise IndexError( message )
    else:
      if z_test.shape[-1] != self.config.len_latent + self.num_classes_gen:
        message = f"This is a generator class-conditioned model. So please make sure to append a one-hot encoded vector\n" + \
                  f"of size {self.num_classes_gen} that indicates the to-be generated sample's class to a latent vector of\n" + \
                  f"size {self.config.len_latent}. Total input size must therefore be {self.config.len_latent + self.num_classes_gen}."
        raise IndexError( message )

    # z_test = z_test.to( self.config.dev )
    x_test = self.gen_model_lagged( z_test ).squeeze() if time_average else self.gen_model( z_test ).squeeze()

    if label is not None:
      print( f'Label Index for Generated Image: {label}' )

    logger = logging.getLogger()
    _old_level = logger.level
    logger.setLevel( 100 )  # ignores potential "clipping input data" warning
    plt.imshow( ( ( ( x_test ) \
                          .cpu().detach() * self.ds_std ) + self.ds_mean ) \
                          .numpy().transpose( 1, 2, 0 ), interpolation = 'none'
    )
    logger.setLevel( _old_level )
    plt.show()

  @torch.no_grad()
  def make_image_grid( self, zs, labels = None, time_average = True, save_path = None ):
    """Generates grid of images from input latent codes, whose size is `np.sqrt( len( zs ) )`."""
    if self.ds_mean is None or self.ds_std is None:
      raise ValueError( "This model does not hold any information about your dataset's mean and/or std.\n" + \
                        "Please provide these (either from your current data configuration or from your pretrained model)." )

    if not zs.dim() == 2:
      raise IndexError( 'Incorrect dimensions of input latent vector. Must be `dim == 2`.' )
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

    if np.sqrt( len( zs ) ) % 1 != 0:
      raise ValueError( 'Argument `zs` must be a perfect square-length in order to make image grid.' )

    sz = int( np.sqrt( len( zs ) ) )
    fig = plt.figure( figsize = ( 8, 8 if labels is None else 9, ) )
    axs = [ fig.add_subplot( sz, sz, i + 1 ) for i in range( sz**2 ) ]
    logger = logging.getLogger()
    _old_level = logger.level
    logger.setLevel( 100 )  # ignores potential "clipping input data" warning
    for n, ax in enumerate( axs ):
      x = self.gen_model_lagged( zs[ n ] ).squeeze() if time_average else self.gen_model( zs[ n ] ).squeeze()
      ax.imshow(
        ( ( x.cpu().detach() * self.ds_std ) + self.ds_mean ).numpy().transpose( 1, 2, 0 ),
        interpolation = 'none'
      )
      if labels is not None:
        ax.set_title( str( labels[ n ].item() ) )
      ax.axis( 'off' )
      ax.set_aspect( 'equal' )
    logger.setLevel( _old_level )
    fig.subplots_adjust( left = 0, right = 1, bottom = 0, top = 1, wspace = 0, hspace = 0 )

    # maintain resolution of images and save
    if save_path is not None:
      bbox = axs[-1].get_window_extent().transformed( fig.dpi_scale_trans.inverted() )
      dpi = ( self.gen_model_lagged.curr_res if time_average else self.gen_model.curr_res ) / bbox.height
      fig.savefig( save_path, dpi = dpi, pad_inches = 0 )

    return ( fig, axs, )

  # .......................................................................... #

  def save_model( self, save_path:Path ):
    if self.not_trained_yet:
      raise Exception( 'Please train your model for atleast 1 iteration before saving.' )

    warnings.filterwarnings( 'ignore', category = UserWarning )

    self.gen_model_metadata = { 'gen_model_upsampler': self.gen_model_upsampler,
                                'num_classes_gen': self.num_classes_gen }
    self.disc_model_metadata = { 'disc_model_downsampler': self.disc_model_downsampler,
                                 'num_classes_disc': self.num_classes_disc }

    scheduler_state_dict_save_args = {
      'scheduler_gen_state_dict': self.scheduler_gen.state_dict() if self.sched_bool else None,
      'scheduler_disc_state_dict': self.scheduler_disc.state_dict() if self.sched_bool else None
    }

    save_path = Path( save_path )
    save_path.parents[0].mkdir( parents = True, exist_ok = True )

    torch.save( {
                'config': self.config,
                'curr_res': self.gen_model.curr_res,
                'alpha': self.gen_model.alpha,
                'gen_model_metadata': self.gen_model_metadata,
                'gen_model_state_dict': self.gen_model.state_dict(),
                'gen_model_lagged_state_dict': self.gen_model_lagged.state_dict() if self.config.use_ewma_gen else None,
                'disc_model_metadata': self.disc_model_metadata,
                'disc_model_state_dict': self.disc_model.state_dict(),
                'nl': self.nl,
                'sched_stop_step': self.sched_stop_step,
                'lr_sched': self.lr_sched,
                **scheduler_state_dict_save_args,
                'optimizer': self.optimizer,
                'opt_gen_state_dict': self.opt_gen.state_dict(),
                'opt_disc_state_dict': self.opt_disc.state_dict(),
                'loss': self.loss,
                'gradient_penalty': self.gradient_penalty,
                'batch_size': self.batch_size,
                'curr_dataset_batch_num': self.curr_dataset_batch_num,
                'curr_epoch_num': self.curr_epoch_num,
                'tot_num_epochs': self.tot_num_epochs,
                'dataset_sz': self.dataset_sz,
                'ac': self.ac,
                'cond_gen': self.cond_gen,
                'cond_disc': self.cond_disc,
                'valid_z': self.valid_z.to( 'cpu' ),
                'valid_label': self.valid_label,
                'grid_inputs_constructed': self.grid_inputs_constructed,
                'rand_idxs': self.rand_idxs,
                'gen_metrics_num': self.gen_metrics_num,
                'disc_metrics_num': self.disc_metrics_num,
                'curr_img_num': self.curr_img_num,
                'nimg_transition_lst': self.nimg_transition_lst,
                'not_trained_yet': self.not_trained_yet,
                'ds_mean': self.ds_mean,
                'ds_std': self.ds_std,
                'latent_distribution': self.latent_distribution,
                'curr_phase_num': self.curr_phase_num,
                'lagged_params': self.lagged_params,
                'progressively_grow': self.progressively_grow }, save_path
    )

  def load_model( self, load_path, dev_of_saved_model = 'cpu' ):
    dev_of_saved_model = dev_of_saved_model.casefold()
    assert ( dev_of_saved_model == 'cpu' or dev_of_saved_model == 'cuda' )
    _map_location = lambda storage,loc: storage if dev_of_saved_model == 'cpu' else None

    checkpoint = torch.load( load_path, map_location = _map_location )

    self.config = checkpoint[ 'config' ]

    self.nl = checkpoint[ 'nl' ]

    # Load pretrained neural networks:
    global ProGAN, ProGenerator, ProDiscriminator
    ProGAN = type( 'ProGAN', ( nn.Module, ABC, ), dict( ProGAN.__dict__ ) )
    ProGAN.reset_state( )

    ProGenerator = type( 'ProGenerator', ( ProGAN, ), dict( ProGenerator.__dict__ ) )
    self.gen_model_metadata = checkpoint[ 'gen_model_metadata' ]
    self.gen_model_upsampler = self.gen_model_metadata[ 'gen_model_upsampler' ]
    self.num_classes_gen = self.gen_model_metadata[ 'num_classes_gen' ]
    self.gen_model = ProGenerator(
      final_res = self.config.res_samples,
      len_latent = self.config.len_latent,
      upsampler = self.gen_model_upsampler,
      blur_type = self.config.blur_type,
      nl = self.nl,
      num_classes = self.num_classes_gen,
      equalized_lr = self.config.use_equalized_lr,
      normalize_z = self.config.normalize_z,
      use_pixelnorm = self.config.use_pixelnorm
    )

    ProDiscriminator = type( 'ProDiscriminator', ( ProGAN, ), dict( ProDiscriminator.__dict__ ) )
    self.disc_model_metadata = checkpoint[ 'disc_model_metadata' ]
    self.disc_model_downsampler = self.disc_model_metadata[ 'disc_model_downsampler' ]
    self.num_classes_disc = self.disc_model_metadata[ 'num_classes_disc' ]
    self.disc_model = ProDiscriminator(
      final_res = self.config.res_samples,
      pooler = self.disc_model_downsampler,
      blur_type = self.config.blur_type,
      nl = self.nl,
      num_classes = self.num_classes_disc,
      equalized_lr = self.config.use_equalized_lr,
      mbstd_group_size = self.config.mbstd_group_size
    )

    assert self.config.init_res <= self.config.res_samples

    # If pretrained model started at a higher resolution than 4:
    _curr_res = checkpoint[ 'curr_res' ]
    if _curr_res > 4:
      _init_res_log2 = int( np.log2( _curr_res ) )
      if float( _curr_res ) != 2**_init_res_log2:
        raise ValueError( 'Only resolutions that are powers of 2 are supported.' )
      num_scale_inc = _init_res_log2 - 2
      for _ in range( num_scale_inc ):
        self.gen_model.increase_scale()
        self.disc_model.increase_scale()

    # this applies to both networks simultaneously and takes care of fade_in_phase
    self.gen_model.alpha = checkpoint[ 'alpha' ]

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
    self.gen_model.load_state_dict( checkpoint[ 'gen_model_state_dict' ] )
    self.gen_model.zero_grad()
    if self.config.use_ewma_gen:
      self.gen_model_lagged.load_state_dict( checkpoint[ 'gen_model_lagged_state_dict' ] )
      self.gen_model_lagged.to( self.config.metrics_dev )
      self.gen_model_lagged.zero_grad()
    self.disc_model.to( self.config.dev )
    self.disc_model.load_state_dict( checkpoint[ 'disc_model_state_dict' ] )
    self.disc_model.zero_grad()

    self.sched_stop_step = checkpoint[ 'sched_stop_step' ]
    self.lr_sched = checkpoint[ 'lr_sched' ]
    self._scheduler_gen_state_dict = checkpoint['scheduler_gen_state_dict']
    self._scheduler_disc_state_dict = checkpoint['scheduler_disc_state_dict']
    self._sched_state_dict_set = False

    self.optimizer = checkpoint['optimizer']
    self.opt_gen.load_state_dict( checkpoint['opt_gen_state_dict'] )
    self.opt_disc.load_state_dict( checkpoint['opt_disc_state_dict'] )

    self.batch_size = checkpoint['batch_size']

    self.loss = checkpoint['loss']

    self.gradient_penalty = checkpoint['gradient_penalty']

    self.curr_dataset_batch_num = checkpoint['curr_dataset_batch_num']
    self.curr_epoch_num = checkpoint['curr_epoch_num']
    self.tot_num_epochs = checkpoint['tot_num_epochs']
    self.dataset_sz = checkpoint['dataset_sz']

    self.ac = checkpoint['ac']
    self.cond_gen = checkpoint['cond_gen']
    self._tensor = torch.FloatTensor
    if self.config.dev == torch.device( 'cuda' ):
      self._tensor = torch.cuda.FloatTensor
    if self.cond_gen:
      self.labels_one_hot_disc = self._tensor( self.batch_size, self.num_classes )
      self.labels_one_hot_gen = self._tensor( self.batch_size * self.config.gen_bs_mult, self.num_classes )
    self.cond_disc = checkpoint['cond_disc']

    self.valid_z = checkpoint['valid_z'].to( self.config.metrics_dev )
    self.valid_label = checkpoint['valid_label']
    if self.valid_label is not None:
      self.valid_label = self.valid_label.to( 'cpu' )
    self.grid_inputs_constructed = checkpoint['grid_inputs_constructed']
    self.rand_idxs = checkpoint['rand_idxs']
    self.gen_metrics_num = checkpoint['gen_metrics_num']
    self.disc_metrics_num = checkpoint['disc_metrics_num']

    self.curr_img_num = checkpoint[ 'curr_img_num' ]
    self.nimg_transition_lst = checkpoint[ 'nimg_transition_lst' ]

    self.not_trained_yet = checkpoint['not_trained_yet']

    # By default, use the pretrained model's statistics (you can change this by changing ds_mean and ds_std manually)
    self.ds_mean = checkpoint['ds_mean']
    self.ds_std = checkpoint['ds_std']
    if self._is_data_configed:
      self.data_config.ds_mean = self.ds_mean.squeeze().tolist()
      self.data_config.ds_std = self.ds_std.squeeze().tolist()

    self.latent_distribution = checkpoint[ 'latent_distribution' ]
    self.curr_phase_num = checkpoint[ 'curr_phase_num' ]
    self.eps = False
    if self.config.eps_drift > 0:
      self.eps = True
    self.lagged_params = checkpoint[ 'lagged_params' ]
    self._progressively_grow = checkpoint[ 'progressively_grow' ]


    self.pretrained_model = True

    # Print configuration:
    print( '---------- Loaded Model Configuration -----------' )
    print( self.config )
    print( '-------------------------------------------------' )
    print( "\n  If you would like to change any of the above configurations,\n" + \
           "  please do so via setting the attributes of your instantiated ProGANLearner().config object.\n" )