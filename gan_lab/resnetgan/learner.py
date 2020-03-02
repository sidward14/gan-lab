# -*- coding: UTF-8 -*-

"""Learner for all ResNet GANs and non-progressive GANs in general.

  Typical usage example:

  First configure your desired GAN on the command-line:
    go to root directory...
    $ python config.py resnetgan
    $ python data_config.py LSUN-Bedrooms path/to/datasets/lsun_bedrooms

  Then write a custom script (or use train.py):
    from gan_lab import get_current_configuration
    from gan_lab.utils.data_utils import prepare_dataset, prepare_dataloader
    from gan_lab.resnetgan.learner import GANLearner

    # get most recent configurations:
    config = get_current_configuration( 'config' )
    data_config = get_current_configuration( 'data_config' )

    # get DataLoader(s)
    train_ds, valid_ds = prepare_dataset( data_config )
    train_dl, valid_dl, z_valid_dl = prepare_dataloader( config, data_config, train_ds, valid_ds )

    # instantiate GANLearner and train:
    learner = GANLearner( config )
    learner.train( train_dl, valid_dl, z_valid_dl )   # train for config.num_main_iters iterations
    learner.config.num_main_iters = 300000            # this is one example of changing your instantiated learner's configurations
    learner.train( train_dl, valid_dl, z_valid_dl )   # train for another 300000 iterations

Note that the above custom script is just a more flexible alternative to running
train.py (you can, for example, run the above on a Jupyter Notebook). You can
always just run train.py.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from .architectures import Generator32PixResnet, \
                           Discriminator32PixResnet, \
                           DiscriminatorAC32PixResnet, \
                           Generator64PixResnet, \
                           Discriminator64PixResnet, \
                           DiscriminatorAC64PixResnet, \
                           FMAP_G, FMAP_D
from _int import get_current_configuration, LearnerConfigCopy, FMAP_SAMPLES
# from progan.architectures import ProDiscriminator
from utils.latent_utils import gen_rand_latent_vars
from utils.backprop_utils import wasserstein_distance_gen, \
                                 nonsaturating_loss_gen, \
                                 minimax_loss_gen, \
                                 wasserstein_distance_disc, \
                                 minimax_loss_disc, \
                                 calc_gp, \
                                 configure_adam_for_gan
from utils.custom_layers import NearestPool2d, BilinearPool2d

import copy
import logging
import warnings
from pathlib import Path
from functools import partial
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update( { 'figure.max_open_warning': 0 } )
import torch
from torch import nn
import torch.nn.functional as F

# from tqdm import tqdm

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

NONREDEFINABLE_ATTRS = ( 'model', 'res_samples', 'res_dataset', 'len_latent',
                         'num_classes', 'class_condition', 'use_auxiliary_classifier',
                         'model_upsample_type', 'model_downsample_type', 'align_corners',
                         'blur_type', 'nonlinearity', 'use_equalized_lr', )

REDEFINABLE_FROM_LEARNER_ATTRS = ( 'batch_size', 'loss', 'gradient_penalty',
                                   'optimizer', 'lr_sched', )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class GANLearner( object ):
  """Base GAN Learner for non=progressive GAN architectures (i.e. not ProGAN or StyleGAN).

  Once instantiated, the GANLearner object's configuration can be changed, but only via
  its self.config attribute (i.e. running 'python config.py [model]' post-instantiation
  will not affect this learner's configuration).
  """
  def __init__( self, config ):
    super( GANLearner, self ).__init__()

    self._model = config.model

    # Set equal to `True` if `self.load_model()` method is called after initialization
    self.pretrained_model = False

    _model_selected = False
    if self.model == 'ResNet GAN':

      # If you want to change an attribute in an already-instantiated GANLearner's config or data_config,
      # change self.config (below) instead of config and self.data_config (also below) instead of data_config:
      self.config = LearnerConfigCopy( config,
                                       self.__class__.__name__,
                                       NONREDEFINABLE_ATTRS,
                                       REDEFINABLE_FROM_LEARNER_ATTRS )
      # pretrained models loaded later on for evaluation should not require data_config.py to have been run:
      self._is_data_configed = False; self._stats_set = False
      self._update_data_config( raise_exception = False )

      _model_selected = True

    if _model_selected:
      self.batch_size = self.config.batch_size
    self.curr_dataset_batch_num = 0
    self.curr_epoch_num = 0
    self.curr_valid_num = 0

    # Supervised/Unsupervised method selection:
    self.num_classes = 0
    self.cond_gen = False
    self.cond_disc = False
    self.ac = False
    self.num_classes_gen = 0
    self.num_classes_disc = 0
    if config.use_auxiliary_classifier or config.class_condition:
      self.num_classes = config.num_classes
      self.num_classes_disc = config.num_classes
      if config.use_auxiliary_classifier:
        self.ac = True
      if config.class_condition:
        self.cond_disc = True if not self.ac else False
        self.num_classes_gen = config.num_classes
        self.cond_gen = True

    assert config.res_samples <= config.res_dataset

    # Generator Upsampling selection:
    if config.model_upsample_type.casefold() == 'nearest':
      _align_corners = None
    elif config.model_upsample_type.casefold() == 'bilinear':
      _align_corners = config.align_corners
    else:
      raise ValueError( "config does not support this model_upsample_type.\n" + \
                        "Supported Upsampling Types are: [ 'nearest', 'bilinear' ]" )
    self.gen_model_upsampler = nn.Upsample(
      scale_factor = 2,
      mode = config.model_upsample_type.casefold(),
      align_corners = _align_corners
    )
    # Discriminator Downsampling selection:
    if config.model_downsample_type.casefold() in ( 'average', 'box', ):
      self.disc_model_downsampler = nn.AvgPool2d(
        kernel_size = 2,
        stride = 2
      )
    elif config.model_downsample_type.casefold() == 'nearest':
      self.disc_model_downsampler = NearestPool2d( )
    elif config.model_downsample_type.casefold() == 'bilinear':
      self.disc_model_downsampler = BilinearPool2d(
        align_corners = config.align_corners
      )
    else:
      raise ValueError( "config does not support this model_downsample_type.\n" + \
                        "Supported Downsampling Types are: [ 'nearest', 'average', 'box', 'bilinear' ]" )

    # Nonlinearity selection:
    if config.nonlinearity.casefold() == 'leaky relu':
      self.nl = nn.LeakyReLU( negative_slope = config.leakiness )
    elif config.nonlinearity.casefold() == 'relu':
      self.nl = nn.ReLU()
    elif config.nonlinearity.casefold() == 'tanh':
      self.nl = nn.Tanh()

    # Architecture selection:
    if _model_selected:
      if self.config.res_samples == 64:
        _gen_model = Generator64PixResnet
        _disc_model = DiscriminatorAC64PixResnet if self.ac else Discriminator64PixResnet
        _fmap_g = FMAP_G; _fmap_d = FMAP_D
      elif self.config.res_samples == 32:
        _gen_model = Generator32PixResnet
        _disc_model = DiscriminatorAC32PixResnet if self.ac else Discriminator32PixResnet
        _fmap_g = FMAP_G*2; _fmap_d = FMAP_D*2
      # TODO: Implement other kinds of GANs that are not ResNet-style GAN, ProGAN, or StyleGAN:
      # elif self.model in ( '___', '___' ):
      #   pass
      else:
        message = 'GANLearner currently only supports 32 pixel and 64 pixel GAN architectures.\n' + \
                  'If a different generated sample resolution is desired, please use the\n' + \
                  'ProGAN or StyleGAN models featured in this package instead.'
        raise ValueError( message )

    # Instantiate Neural Networks:
    self.gen_model = None
    self.disc_model = None
    if _model_selected:
      self.gen_model = _gen_model(
        len_latent = self.config.len_latent,
        fmap = _fmap_g,
        upsampler = self.gen_model_upsampler,
        blur_type = self.config.blur_type,
        nl = self.nl,
        num_classes = self.num_classes_gen,
        equalized_lr = self.config.use_equalized_lr
      )

      self.disc_model = _disc_model(
        fmap = _fmap_d,
        pooler = self.disc_model_downsampler,
        blur_type = self.config.blur_type,
        nl = self.nl,
        num_classes = self.num_classes_disc,
        equalized_lr = self.config.use_equalized_lr
      )

      self.gen_model.to( self.config.dev )
      self.disc_model.to( self.config.dev )

      # Generator and Discriminator resolutions must match:
      assert self.gen_model.res == self.disc_model.res

    self._tensor = torch.FloatTensor
    if config.dev == torch.device( 'cuda' ):
      self._tensor = torch.cuda.FloatTensor

    if _model_selected and self.cond_gen:
      self.labels_one_hot_disc = self._tensor( self.batch_size, self.num_classes )
      self.labels_one_hot_gen = self._tensor( self.batch_size * self.config.gen_bs_mult, self.num_classes )

    # Loss Function:
    if _model_selected:
      self._loss = config.loss.casefold()
      self._set_loss( )

    # Gradient Regularizer:
    self._gradient_penalty = config.gradient_penalty
    if _model_selected:
      self.gp_func = partial(
        calc_gp,
        gp_type = self._gradient_penalty,
        nn_disc = self.disc_model,
        lda = self.config.lda,
        gamma = self.config.gamma
      )

    # Auxiliary Classifier:
    if self.ac:
      self.loss_func_aux = nn.CrossEntropyLoss( )

    # Optimizer:
    self._optimizer = config.optimizer.casefold()
    self.opt_gen = None
    self.opt_disc = None
    if _model_selected:
      self._set_optimizer( )

    # Learning Rate Scheduler:
    self._lr_sched = None
    self.sched_bool = False
    self.sched_stop_step = None
    self.scheduler_gen = None
    self.scheduler_disc = None
    if config.lr_sched is not None:
      self._lr_sched = config.lr_sched.casefold()
      self.sched_bool = True
      self.sched_stop_step = 0

    # Validation Set-specific Inits:
    self.grid_inputs_constructed = False
    self.rand_idxs = None
    self.valid_z = None
    self.valid_label = None
    self.gen_metrics_num = 0
    self.disc_metrics_num = 0

    self._param_tensor_num = 0

    # Training State data:
    self.curr_img_num = 0

    # Whether to begin training for the first time or continue training:
    self.not_trained_yet = True

    # Print configuration:
    if _model_selected:
      print( '-------- Initialized Model Configuration --------' )
      print( self.config )
      print( '-------------------------------------------------' )
      # print( "  If you would like to alter any of the above configurations,\n" + \
      #        "  please do so via altering your instantiated GANLearner().config's attributes." )
      print( '\n    Ready to train!\n' )


  def forward_prop_batch( self, inputb, fprop_type = 'full' ):
    """In case one wants to do just forward prop."""
    fprop_type = fprop_type.casefold()
    if fprop_type == 'generator':
      return self.gen_model( inputb )
    elif fprop_type == 'discriminator':
      return self.disc_model( inputb )
    elif fprop_type == 'full':
      return self.disc_model( self.gen_model( inputb ) )
    else:
      raise ValueError( 'Invalid forward prop type.' )

  def backprop_batch( self, loss, bprop_type = 'full', opt = None ):
    """In case one wants to do just backprop."""
    if opt is not None:
      loss.backward()
      opt.step()
      opt.zero_grad()

    return loss.item()

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

    self.gen_model.eval()
    self.disc_model.eval()

    metrics_type = metrics_type.casefold()
    if metrics_type not in ( 'generator', 'critic', 'discriminator', ):
      raise Exception( 'Invalid metrics_type. Only "generator", "critic", or "discriminator" are accepted.' )
    metrics = [ metric.casefold() for metric in metrics ]

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
                                 device = self.config.dev, dtype = torch.float32 ) for metric in metrics }

    if z_valid_dl is not None:
      for n in range( _len_z_valid_dl ):
        # Uncomment the below if validation set is taking up too much memory
        # zb = gen_rand_latent_vars( num_samples = self.batch_size, length = config.len_latent, distribution = 'normal', device = self.config.dev )
        zbatch = next( z_valid_dataiter )
        zb = ( zbatch[0] ).to( self.config.dev )
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
              self.valid_z = torch.FloatTensor( self.config.img_grid_sz**2, zb.shape[1] ).to( self.config.dev )
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
              fig, _ = self.make_image_grid( zs = self.valid_z, labels = self.valid_label )
              save_img_grid_dir = \
                self.config.save_samples_dir/self.model.casefold().replace( " ", "" )/self.data_config.dataset/'image_grid'
              save_img_grid_dir.mkdir( parents = True, exist_ok = True )
              fig.savefig( save_img_grid_dir/( str( self.gen_metrics_num ) + '.png' ),
                           bbox_inches = 'tight', pad_inches = 0 )
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
            xb = ( xbatch[0] ).to( self.config.dev )
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
    metrics_vals = [
      ( vals_tensor.sum() / _len_z_valid_ds ).item() \
      for vals_tensor in metrics_tensors.values()
    ]
    metrics_vals = [ f'{metric}:{metrics_vals[n]}' for n, metric in enumerate( metrics ) if metric != 'image grid' ]

    self.gen_model.train()
    self.disc_model.train()

    return metrics_vals

  def train( self, train_dl, valid_dl = None, z_valid_dl = None,
             num_main_iters = None, num_gen_iters = None, num_disc_iters = None ):

    """Efficient & fast implementation of GAN training (at the expense of some messy/repetitive code).

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

    self.gen_model.to( self.config.dev )
    self.gen_model.train()
    self.disc_model.to( self.config.dev )
    self.disc_model.train()

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
      print( 'STARTING FROM ITERATION 0:' )
      self.train_dataiter = iter( train_dl )
    else:
      print( 'CONTINUING FROM WHERE YOU LEFT OFF:' )
      if self.pretrained_model:
        self.train_dataiter = iter( train_dl )

    # ------------------------------------------------------------------------ #

    # for itr in tqdm( range( num_main_iters ) ):
    for itr in range( num_main_iters ):
      print( 'ITERATION #', itr )

      if self.sched_bool:
        with warnings.catch_warnings():
          warnings.simplefilter( 'ignore' )
          print( f'Generator LR: {self.scheduler_gen.get_lr()} |', \
                 f'Discriminator LR: {self.scheduler_disc.get_lr()}'
          )

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
                                     distribution = self.config.latent_distribution, device = self.config.dev )
          self.labels_one_hot_gen.zero_()
          self.labels_one_hot_gen.scatter_( 1, gen_labels, 1 )
          if self.ac: gen_labels.squeeze_()
          zb = torch.cat( ( zb, self.labels_one_hot_gen, ), dim = 1 )
        else:
          zb = gen_rand_latent_vars( num_samples = self.batch_size * self.config.gen_bs_mult, length = self.config.len_latent,
                                     distribution = self.config.latent_distribution, device = self.config.dev )
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

        # Compute metrics for generator (validation metrics should be for entire validation set):
        metrics_vals = []
        _valid_title = []
        if z_valid_dl is not None and self.config.gen_metrics:
          if ( itr + 1 ) % self.config.num_iters_valid == 0 or itr == 0:
            if gen_iter == 0 and itr != 0:
              end = timer(); print( f'\nTime since last Validation Set: {end - start} seconds.\n' )

            metrics_vals = self.compute_metrics(
              metrics = self.config.gen_metrics, metrics_type = 'Generator',
              z_valid_dl = z_valid_dl, valid_dl = None
            )
            _valid_title = [ '|\n', 'Validation Metrics:' ]

        print( f'Generator Batch Metrics: Batch # {gen_iter} |', \
               f'Train Loss {loss_train_gen.item()}', *_valid_title, *metrics_vals
        )

      #------------------------- TRAIN DISCRIMINATOR ----------------------------

      # these are set to `False` for the Generator because you don't need
      # the Discriminator's parameters' gradients when chain-ruling back to the generator
      for p in self.disc_model.parameters(): p.requires_grad_( True )

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
                                     distribution = self.config.latent_distribution, device = self.config.dev )
          self.labels_one_hot_disc.zero_()
          self.labels_one_hot_disc.scatter_( 1, gen_labels, 1 )
          if self.ac: gen_labels.squeeze_()
          zb = torch.cat( ( zb, self.labels_one_hot_disc, ), dim = 1 )
        else:
          zb = gen_rand_latent_vars( num_samples = self.batch_size, length = self.config.len_latent,
                                     distribution = self.config.latent_distribution, device = self.config.dev )
        with torch.no_grad(): zbv = zb  # makes sure to totally freeze the generator when training discriminator
        _xgenb = self.gen_model( zbv ).detach()

        # Sample real data x:
        batch = next( self.train_dataiter, None )
        if batch is None:
          self.curr_epoch_num += 1
          print( f'\nEPOCH # {self.curr_epoch_num - 1} COMPLETE. BEGIN EPOCH #', \
                 f'{self.curr_epoch_num}\n' )
          self.train_dataiter = iter( train_dl )
          batch = next( self.train_dataiter )
        xb = ( batch[0] ).to( self.config.dev )
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

        # Backprop:
        loss_train_disc.backward()  # compute the gradients
        self.opt_disc.step()        # update the parameters you specified to the optimizer with backprop
        # self.opt_disc.zero_grad()

        # Compute metrics for discriminator (validation metrics should be for entire validation set):
        metrics_vals = []
        _valid_title = []
        if z_valid_dl is not None and valid_dl is not None and self.config.disc_metrics:
          if ( itr + 1 ) % self.config.num_iters_valid == 0 or itr == 0:
            metrics_vals = self.compute_metrics(
              metrics = self.config.disc_metrics, metrics_type = 'Discriminator',
              z_valid_dl = z_valid_dl, valid_dl = valid_dl
            )
            _valid_title = [ '|\n', 'Validation Metrics:' ]

            if disc_iter == num_disc_iters - 1: start = timer()

        print( f'Discriminator Batch Metrics: Batch # {disc_iter} |', \
               f'Train Loss {loss_train_disc.item()}', *_valid_title, *metrics_vals
        )

        self.curr_dataset_batch_num += 1
        self.curr_img_num += self.batch_size

      if self.sched_bool:
        self.scheduler_gen.step()
        self.scheduler_disc.step()

      if self.not_trained_yet:
        self.not_trained_yet = False

      # Save model every self.config.num_iters_save_model iterations:
      if ( itr + 1 ) % self.config.num_iters_save_model == 0:
        self.gen_model.to( 'cpu' )
        self.gen_model.eval()
        self.disc_model.to( 'cpu' )
        self.disc_model.eval()

        self.save_model( self.config.save_model_dir/( self.model.casefold().replace( " ", "" ) + '_model.tar' ) )

        self.gen_model.to( self.config.dev )
        self.gen_model.train()
        self.disc_model.to( self.config.dev )
        self.disc_model.train()

    self.gen_model.to( 'cpu' )
    self.gen_model.eval()
    self.disc_model.to( 'cpu' )
    self.disc_model.eval()

  # .......................................................................... #

  def calc_gp( self, gen_data, real_data ):
    """Method that takes care of all gradient regularizers."""
    if self.gradient_penalty in ( 'wgan-gp', 'r1', ):
      real_data = real_data.view(
        -1, FMAP_SAMPLES, real_data.shape[2], real_data.shape[3]
      )
    if self.gradient_penalty in ( 'wgan-gp', 'r2', ):
      gen_data = gen_data.view(
        -1, FMAP_SAMPLES, gen_data.shape[2], gen_data.shape[3]
      )

    # whether to penalize the discriminator gradients on the real distribution, fake distribution, or an interpolation of both
    # TODO: Implement class-conditioning (in the discriminator) option (Mirza & Osindero, 2014).
    if self.gradient_penalty == 'wgan-gp':
      eps = torch.rand( self.batch_size, 1, 1, 1, device = self.config.dev )
      xb = ( eps * gen_data.detach() + \
           ( 1 - eps ) * real_data.detach() ).to( self.config.dev )
    elif self.gradient_penalty == 'r1':
      xb = real_data.detach().to( self.config.dev )
    elif self.gradient_penalty == 'r2':
      xb = gen_data.detach().to( self.config.dev )

    # now start tracking in the computation graph again
    xb.requires_grad_( True )

    if isinstance( self.disc_model, ( DiscriminatorAC32PixResnet,
                                      DiscriminatorAC64PixResnet, ) ):
      outb, _ = self.disc_model( xb )
    else:
      outb = self.disc_model( xb )
    # print( xb.requires_grad, outb.requires_grad )
    outb_grads = torch.autograd.grad(
      outb,
      xb,
      grad_outputs = torch.ones( self.batch_size ).to( self.config.dev ),
      create_graph = True, retain_graph = True, only_inputs = True )[0]

    if self.gradient_penalty == 'wgan-gp':
      if self.config.gamma != 1.:
        outb_gp = \
          ( ( outb_grads.norm( 2, dim = 1 ) - self.config.gamma )**2 / self.config.gamma**2 ).mean() * self.config.lda
      else:
        outb_gp = \
          ( ( outb_grads.norm( 2, dim = 1 ) - 1. )**2 ).mean() * self.config.lda / 2.
    elif self.gradient_penalty in ( 'r1', 'r2', ):
      outb_gp = ( outb_grads.norm( 2, dim = 1 )**2 ).mean() * self.config.lda / 2.

    return outb_gp

  # .......................................................................... #

  @property
  def lr_sched( self ):
    return self._lr_sched

  @lr_sched.setter
  def lr_sched( self, new_lr_sched ):
    self._lr_sched = None
    self.sched_bool = False
    if not self.pretrained_model:
      self.sched_stop_step = None
    self.scheduler_gen = None
    self.scheduler_disc = None
    if new_lr_sched is not None:
      self._lr_sched = new_lr_sched.casefold()
      self.sched_bool = True
      if not self.pretrained_model:
        self.sched_stop_step = 0

  def _set_scheduler( self ):
    # self.tot_num_epochs = ( self.batch_size * num_main_iters * num_disc_iters * 1. ) / self.dataset_sz
    if self._lr_sched == 'linear decay':
      # self.scheduler_fn = lambda epoch: 1. - ( epoch + self.sched_stop_step ) * ( 1. / ( self.tot_num_epochs//1 ) )
      self.scheduler_fn = lambda main_iter: 1. - ( main_iter + self.sched_stop_step ) * ( 1. / self.num_main_iters )
    elif self._lr_sched == 'custom':
      self.scheduler_fn = eval( self.config.lr_sched_custom )
    # TODO: add more types of LR scheduling
    else:
      raise ValueError( "config does not support this LR scheduler.\n" + \
                        "Currently supported LR Schedulers are: [ 'linear decay', 'custom' ]" )

    self.scheduler_gen = \
      torch.optim.lr_scheduler.LambdaLR( self.opt_gen, self.scheduler_fn, last_epoch = -1 )
    self.scheduler_disc = \
      torch.optim.lr_scheduler.LambdaLR( self.opt_disc, self.scheduler_fn, last_epoch = -1 )

    self._scheduler_gen_state_dict = self.scheduler_gen.state_dict()
    self._scheduler_disc_state_dict = self.scheduler_disc.state_dict()
    if self.pretrained_model and not self._sched_state_dict_set:
      self.scheduler_gen.load_state_dict( self._scheduler_gen_state_dict )
      self.scheduler_disc.load_state_dict( self._scheduler_disc_state_dict )
      self._scheduler_gen_state_dict = self.scheduler_gen.state_dict()
      self._scheduler_disc_state_dict = self.scheduler_disc.state_dict()
      self._sched_state_dict_set = True

  @property
  def optimizer( self ):
    return self._optimizer

  @optimizer.setter
  def optimizer( self, new_optimizer ):
    self._optimizer = new_optimizer.casefold()
    self._set_optimizer( )

  def _set_optimizer( self ):
    if self._optimizer == 'adam':
      adam_gan = configure_adam_for_gan(
        lr_base = self.config.lr_base,
        betas = ( self.config.beta1, self.config.beta2 ),
        eps = self.config.eps,
        wd = self.config.wd
      )
      self.opt_gen = adam_gan( params = self.gen_model.parameters() )
      self.opt_disc = adam_gan( params = self.disc_model.parameters() )
    elif self._optimizer == 'rmsprop':
      raise NotImplementedError( 'RMSprop optimizer not yet implemented.' )  # TODO:
    elif self._optimizer == 'momentum':
      raise NotImplementedError( 'Momentum optimizer not yet implemented.' )  # TODO:
    elif self._optimizer == 'sgd':
      raise NotImplementedError( 'SGD optimizer not yet implemented.' )  # TODO:
    else:
      raise ValueError( "config does not support this optimizer.\n" + \
                        "Supported Optimizers are: [ 'adam', 'rmsprop', 'momentum', 'sgd' ]" )

  @property
  def gradient_penalty( self ):
    return self._gradient_penalty.casefold()

  @gradient_penalty.setter
  def gradient_penalty( self, new_gradient_penalty ):
    self._gradient_penalty = new_gradient_penalty.casefold()
    self.gp_func = partial(
      calc_gp,
      gp_type = self._gradient_penalty,
      nn_disc = self.disc_model,
      lda = self.config.lda,
      gamma = self.config.gamma
    )

  @property
  def loss( self ):
    return self._loss

  @loss.setter
  def loss( self, new_loss ):
    self._loss = new_loss.casefold()
    self._set_loss( )

  def _set_loss( self ):
    if self._loss == 'wgan':
      self.loss_func_gen = wasserstein_distance_gen
      self.loss_func_disc = wasserstein_distance_disc
      self._dummy_target_gen = None
      self._dummy_target_real = None
    elif self._loss in ( 'nonsaturating', 'minimax', ):
      self.loss_func_disc = minimax_loss_disc
      self._dummy_target_gen = torch.zeros( self.batch_size ).to( self.config.dev )
      self._dummy_target_real = torch.ones( self.batch_size ).to( self.config.dev )
      if self._loss == 'nonsaturating':
        self.loss_func_gen = nonsaturating_loss_gen
      elif self._loss == 'minimax':
        self.loss_func_gen = minimax_loss_gen
    else:
      raise ValueError( "config does not support this loss.\n" + \
                        "Currently supported Loss Functions are: [ 'wgan', 'nonsaturating', 'minimax' ]" )

  @property
  def model( self ):
    return self._model

  @model.setter
  def model( self, new_model ):
    message = f"{self.__class__.__name__}().model attribute cannot be changed once {self.__class__.__name__} is instantiated.\n" + \
              f"Instead, please run 'python config.py {new_model}' and then instantiate a new {self.__class__.__name__}."
    raise AttributeError( message )

  # .......................................................................... #

  @torch.no_grad()
  def plot_sample( self, z_test, label = None ):
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
    x_test = self.gen_model( z_test ).squeeze()

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
  def make_image_grid( self, zs, labels = None ):
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
      x = self.gen_model( zs[ n ] ).squeeze()
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

    return ( fig, axs, )

  # .......................................................................... #

  def _update_data_config( self, raise_exception = True ):
    def _set_norm_statistics( ):
      self._stats_set = True
      # By default, use the pretrained model's or the previously-trained model's statistics (you can change this by changing ds_mean and ds_std manually)
      if not self.pretrained_model:
        self.ds_mean = torch.FloatTensor( self.data_config.ds_mean ).unsqueeze( dim = 1 ).unsqueeze( dim = 2 )
        self.ds_std = torch.FloatTensor( self.data_config.ds_std ).unsqueeze( dim = 1 ).unsqueeze( dim = 2 )
      self.data_config.ds_mean = self.ds_mean.squeeze().tolist()
      self.data_config.ds_std = self.ds_std.squeeze().tolist()

    if not self._is_data_configed and not self.pretrained_model:
      self.ds_mean = None
      self.ds_std = None
    if self._is_data_configed or self._stats_set:
      _set_norm_statistics( )

    self.data_config = get_current_configuration( 'data_config', raise_exception = raise_exception )
    self._is_data_configed = True if self.data_config is not None else False
    # self.num_datasets_used += 1 if self.data_config is not None else 0

    if self._is_data_configed and not self._stats_set:
      _set_norm_statistics( )

  # .......................................................................... #

  def save_model( self, save_path:Path ):
    if self.not_trained_yet:
      raise Exception( 'Please train your model for atleast 1 iteration before saving.' )

    if self.config.res_samples == 64:
      _fmap_g = FMAP_G; _fmap_d = FMAP_D
    elif self.config.res_samples == 32:
      _fmap_g = FMAP_G*2; _fmap_d = FMAP_D*2

    self.gen_model_metadata = { 'gen_model': self.gen_model.__class__,
                                'fmap_g': _fmap_g,
                                'gen_model_upsampler': self.gen_model_upsampler,
                                'num_classes_gen': self.num_classes_gen }
    self.disc_model_metadata = { 'disc_model': self.disc_model.__class__,
                                 'fmap_d': _fmap_d,
                                 'disc_model_downsampler': self.disc_model_downsampler,
                                 'num_classes_disc': self.num_classes_disc }

    scheduler_state_dict_save_args = {
      'scheduler_gen_state_dict': self.scheduler_gen.state_dict() if self.sched_bool else None,
      'scheduler_disc_state_dict': self.scheduler_disc.state_dict() if self.sched_bool else None
    }

    save_path = Path( save_path )
    save_path.parents[0].mkdir( parents = True, exist_ok = True )

    torch.save( {
                'config': self.config,
                'gen_model_metadata': self.gen_model_metadata,
                'gen_model_state_dict': self.gen_model.state_dict(),
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
                'curr_valid_num': self.curr_valid_num,
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
                'not_trained_yet': self.not_trained_yet,
                'ds_mean': self.ds_mean,
                'ds_std': self.ds_std }, save_path
    )

  def load_model( self, load_path, dev_of_saved_model = 'cpu' ):
    dev_of_saved_model = dev_of_saved_model.casefold()
    assert ( dev_of_saved_model == 'cpu' or dev_of_saved_model == 'cuda' )
    _map_location = lambda storage,loc: storage if dev_of_saved_model == 'cpu' else None

    checkpoint = torch.load( load_path, map_location = _map_location )

    self.config = checkpoint[ 'config' ]

    self.nl = checkpoint[ 'nl' ]

    # Load pretrained neural networks:
    self.gen_model_metadata = checkpoint[ 'gen_model_metadata' ]
    _fmap_g = self.gen_model_metadata[ 'fmap_g' ]
    self.gen_model_upsampler = self.gen_model_metadata[ 'gen_model_upsampler' ]
    self.num_classes_gen = self.gen_model_metadata[ 'num_classes_gen' ]
    self.gen_model = self.gen_model_metadata[ 'gen_model' ](
      len_latent = self.config.len_latent,
      fmap = _fmap_g,
      upsampler = self.gen_model_upsampler,
      blur_type = self.config.blur_type,
      nl = self.nl,
      num_classes = self.num_classes_gen,
      equalized_lr = self.config.use_equalized_lr
    )
    self.gen_model.to( self.config.dev )
    self.gen_model.load_state_dict( checkpoint[ 'gen_model_state_dict' ] )
    self.gen_model.zero_grad()

    self.disc_model_metadata = checkpoint[ 'disc_model_metadata' ]
    _fmap_d = self.disc_model_metadata[ 'fmap_d' ]
    self.disc_model_downsampler = self.disc_model_metadata[ 'disc_model_downsampler' ]
    self.num_classes_disc = self.disc_model_metadata[ 'num_classes_disc' ]
    self.disc_model = self.disc_model_metadata[ 'disc_model' ](
      fmap = _fmap_d,
      pooler = self.disc_model_downsampler,
      blur_type = self.config.blur_type,
      nl = self.nl,
      num_classes = self.num_classes_disc,
      equalized_lr = self.config.use_equalized_lr
    )
    self.disc_model.to( self.config.dev )
    self.disc_model.load_state_dict( checkpoint[ 'disc_model_state_dict' ] )
    self.disc_model.zero_grad()

    # Generator and Discriminator resolutions must match:
    assert self.gen_model.res == self.disc_model.res

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
    self.curr_valid_num = checkpoint['curr_valid_num']
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

    self.valid_z = checkpoint['valid_z'].to( self.config.dev )
    self.valid_label = checkpoint['valid_label']
    if self.valid_label is not None:
      self.valid_label = self.valid_label.to( 'cpu' )
    self.grid_inputs_constructed = checkpoint['grid_inputs_constructed']
    self.rand_idxs = checkpoint['rand_idxs']
    self.gen_metrics_num = checkpoint['gen_metrics_num']
    self.disc_metrics_num = checkpoint['disc_metrics_num']

    self.curr_img_num = checkpoint[ 'curr_img_num' ]

    self.not_trained_yet = checkpoint['not_trained_yet']

    # By default, use the pretrained model's statistics (you can change this by changing ds_mean and ds_std manually)
    self.ds_mean = checkpoint['ds_mean']
    self.ds_std = checkpoint['ds_std']
    if self._is_data_configed:
      self.data_config.ds_mean = self.ds_mean.squeeze().tolist()
      self.data_config.ds_std = self.ds_std.squeeze().tolist()


    self.pretrained_model = True

    # Print configuration:
    print( '---------- Loaded Model Configuration -----------' )
    print( self.config )
    print( '-------------------------------------------------' )
    print( "\n  If you would like to change any of the above configurations,\n" + \
           "  please do so via setting the attributes of your instantiated GANLearner().config object.\n" )