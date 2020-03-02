# -*- coding: UTF-8 -*-

"""Configures package for the user-specified model type and hyperparameters.

Run 'python config.py [resnet, progan, stylegan]' on the command-line to
configure (or re-configure) this package for the model type you just specified,
and optionally specify any set of '--[keyword argument]' to override any default
hyperparameters (the defaults work well).

  Typical usage examples:

  $ python config.py stylegan  [OR]
  $ python config.py progan --dev=cuda --enable_cudnn_autotuner --init_res=8

After running this module, you will be able to instantiate (or re-instantiate)
a Learner corresponding to the model type & hyperparameters that you just
configured (or re-configured) for, at any later time.

If you would like to see a list of what each argument does,
run 'python config.py [resnet, progan, stylegan] -h' on the command-line.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from _int import str2bool, get_current_configuration

import os
import sys
import argparse
from functools import partial
# import time
from pathlib import Path
import pickle

import numpy as np
import torch
from torchvision import transforms

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

BS = 64
NIMG_TRANSITION = 600000

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

if __name__ == '__main__':

  parser = argparse.ArgumentParser( prog = 'GAN Lab', description = 'StyleGAN, ProGAN, and ResNet GANs (model configuration)' )

  # ---------------------------------------------------------------------------- #

  # POSITIONAL (REQUIRED) ARGUMENTS:
  # --------------------------------

  parser.add_argument( 'model', type = str, help = 'which of the 3 models (ResNet GAN, ProGAN, or StyleGAN) to use' )


  if len( sys.argv ) < 2:
    raise TypeError( 'config.py requires atleast 1 argument (the model type), and then optional keyword arguments (prefixed with "--").\n' + \
                     ' Example: "$ python config.py stylegan --init_res=8"' )

  if sys.argv[1].casefold() in ( 'resnet', 'res-net', 'res net', 'resnet gan',
                                 'res-net gan', 'res net gan', 'resnetgan', ):
    _model_type = 'ResNet GAN'
  elif sys.argv[1].casefold() in ( 'progan', 'pro-gan', 'pro gan', 'pggan', 'pg-gan', 'pg gan',
                                   'progressivegan', 'progressive-gan', 'progressive gan', ):
    _model_type = 'ProGAN'
  elif sys.argv[1].casefold() in ( 'stylegan', 'style-gan', 'style gan', ):
    _model_type = 'StyleGAN'
  else:
    raise ValueError( 'Invalid model input. The GAN Lab currently only' + \
                      ' supports ResNet GAN, Progressive GAN, or StyleGAN.' )

  # ---------------------------------------------------------------------------- #

  # KEYWORD (OPTIONAL) ARGUMENTS:
  # -----------------------------

  parser.add_argument(
    '--dev',
    type = str.casefold,
    default = DEV,
    choices = [ 'cpu', 'cuda' ],
    help = 'device for memory allocation for the generator and discriminator'
  )
  # TODO: Implement multi-GPU support
  parser.add_argument( '--n_gpu', default = 1, type = int, help = "number of GPUs to use; ignored if config.dev is 'cpu'" )
  # TODO: Implement dynamic batch size control of _MultiProcessDataLoaderIter instances (instantiated when num_workers > 0).
  parser.add_argument( '--enable_cudnn_autotuner', type = str2bool, nargs = '?', const = True, default = False, \
    help = "whether to enable CuDNN auto-tuner to find best algorithm to use for your hardware; usually leads to faster runtime" + \
           " when input size for neural network doesn't vary, as then cuddn will not need to benchmark every time input size changes" )

  parser.add_argument( '--random_seed', type = int, default = -1, \
    help = 'seed for RNG operations such as latent vector sampling, noise, etc.; making this a non-negative integer (instead of -1)' + \
           ' leads to deterministic output, which may be desired if a goal is reproducibility; making this -1 randomly samples the seed instead' )

  parser.add_argument( '--gen_bs_mult', type = int, default = 1, \
    help = 'multiplicative factor for training generator with more data per batch than discriminator;' + \
           ' some datasets (such as CIFAR-10 in ResNet GAN) benefit from this' )

  parser.add_argument( '--num_gen_iters', type = int, default = 1, \
    help = 'number of generator iterations in one main iteration through generator and discriminator' )

  parser.add_argument( '--loss', type = str.casefold, default = 'wgan', choices = [ 'wgan', 'nonsaturating', 'minimax' ], \
    help = 'type of loss function to use for the generator and discriminator' )
  parser.add_argument( '--gradient_penalty', type = str.casefold, default = 'wgan-gp', choices = [ None, 'wgan-gp', 'r1', 'r2' ], \
    help = "type of gradient regularizer to use; options 'R1' and 'R2' are the zero-centered gradient regularizers from" + \
           " `Which Training Methods for GANs do actually Converge?` (Mescheder et al., 2018)" )
  parser.add_argument( '--lda', type = float, default = 10., \
    help = 'gradient penalty multiplicative coefficient (for the `wgan-gp`, `r1`, and `r2` --gradient_penalty options above)' )
  parser.add_argument( '--gamma', type = float, default = 1., \
    help = "gradient penalty degree of Lipschitz constraint; multiclass datasets may benefit from gamma > 1. to help mitigate ghosting;" + \
           " ignored if --gradient_penalty is not set to 'wgan-gp'" )

  parser.add_argument( '--lr_sched_custom', type = str.casefold, default = None, \
    help = 'learning rate scheduler for custom --lr_sched argument; otherwise ignored; this should be a lambda function string' )

  parser.add_argument( '--optimizer', type = str.casefold, default = 'adam', \
    choices = [ 'adam', 'rmsprop', 'momentum', 'sgd' ], help = 'which optimizer to train with' )
  parser.add_argument( '--beta1', type = float, default = 0., help = 'momentum EWMA decay coefficient (default value recommended for Adam)' )
  parser.add_argument( '--eps', type = float, default = 1.e-8, help = 'to prevent division by 0 (default value recommended for Adam)' )
  parser.add_argument( '--wd', type = float, default = 0., help = 'weight decay multiplicative coefficient' )

  parser.add_argument( '--align_corners', type = str2bool, nargs = '?', const = True, default = False, \
    help = 'only applies for bilinear; if `True`, aligns corner pixels of input and output tensors of interpolation' )
  parser.add_argument( '--model_upsample_type', type = str.casefold, default = 'nearest', choices = [ 'nearest', 'bilinear' ], \
    help = 'type of upsampling that the generator conducts to go from latent space to image space' )
  parser.add_argument( '--model_downsample_type', type = str.casefold, default = 'average', choices = [ 'nearest', 'average', 'box', 'bilinear' ], \
    help = 'type of downsampling that the discriminator conducts to go from image space to final feature space' )

  parser.add_argument( '--latent_distribution', type = str.casefold, default = 'normal', choices = [ 'normal', 'uniform' ], \
    help = 'to sample elements of latent vector from normal distribution or uniform distribution' )

  # TODO: Implement class-conditioning of discriminator when not using auxiliary classifier. Also implement more modern forms of the latter.
  parser.add_argument( '--num_classes', type = int, default = 0, \
    help = 'number of classes used for class-conditioning (if `True` below) and/or auxiliary classifier (if `True` below);' + \
           ' if both are `False` below, the model is unsupervised' )
  parser.add_argument( '--class_condition', type = str2bool, nargs = '?', const = True, default = False, \
    help = 'whether to use class conditioning in generator as well as discriminator (the latter only if not using auxiliary classifier (below))' )
  parser.add_argument( '--use_auxiliary_classifier', type = str2bool, nargs = '?', const = True, default = False, \
    help = 'whether to use auxiliary classifier at the end of discriminator (ACGAN)' )
  parser.add_argument( '--ac_disc_scale', type = float, default = 1., \
    help = 'multiplicative factor applied to ACGAN (if --use_auxiliary_classifier set to `True` above) loss when training the discriminator' )
  parser.add_argument( '--ac_gen_scale', type = float, default = .1, \
    help = 'multiplicative factor applied to ACGAN (if --use_auxiliary_classifier set to `True` above) loss when training the generator' )

  # NOTE: All of the below 6 arguments will be ignored if --include_valid_set in data_config.py is set to `False`:
  parser.add_argument( '--num_iters_valid', type = int, default = 1000, \
    help = 'number of main iterations until evaluating metrics again using validation set during training;' + \
           ' ignored if user does not wish to use a validation set during training (see data_config.py for details)' )
  parser.add_argument( '--metrics_dev', type = str.casefold, default = 'cpu', choices = [ 'cpu', 'cuda' ], \
    help = 'device for memory allocation when computing metrics using the validation set' )
  parser.add_argument( '--gen_metrics', type = list, default = [ 'generator loss', 'fake realness', 'image grid' ], \
    choices = [ 'generator loss', 'fake realness', 'image grid' ], help = 'what metrics to evaluate for the generator' )
  parser.add_argument( '--disc_metrics', type = list, default = [ 'discriminator loss', 'fake realness', 'real realness' ], \
    choices = [ 'discriminator loss', 'fake realness', 'real realness' ], help = 'what metrics to evaluate for the discriminator' )
  parser.add_argument( '--img_grid_sz', type = int, default = 4, \
    help = 'size of image grid saved every time metric evaluation is performed on the validation set;' + \
           ' only utilized if image grid is specified in gen_metrics (2 above)' )
  parser.add_argument( '--img_grid_show_labels', type = str2bool, nargs = '?', const = True, default = True, \
    help = 'whether to show the ground truth labels on the image grid for supervised models; ignored if unsupervised model' + \
           ' (i.e. if both --class_condition and --use_auxiliary_classifier above are `False`)' )

  parser.add_argument( '--save_samples_dir', type = Path, default = os.path.abspath( os.path.dirname( __file__ ) ) + '/samples/', \
    help = "root directory where all saved samples are stored (e.g. the image grid from the 'image grid' metric (see --gen_metrics above))" )

  parser.add_argument( '--num_iters_save_model', type = int, default = 1000, help = "number of main iterations until model is saved again during training" )
  parser.add_argument( '--save_model_dir', type = Path, default = os.path.abspath( os.path.dirname( __file__ ) ) + '/models/', \
    help = "root directory where the model is saved by default, such as every config.num_iters_save_model iterations (see --num_iters_save_model above)" )

  parser.add_argument( '--num_workers', type = int, default = 0, \
    help = 'number of subprocesses to use for data loading; 0 will make data load in main process' )
  parser.add_argument( '--pin_memory', type = str2bool, nargs = '?', const = True, default = True if DEV == 'cuda' else False, \
    help = 'whether the dataloader will copy torch.Tensors into CUDA pinned memory, thus enabling faster data transfer; set to `False` if not using CUDA' )

  # ............................................................................ #

  # ResNet GAN Keyword Arguments:
  if _model_type is 'ResNet GAN':
    parser.add_argument( '--batch_size', type = int, default = BS, help = 'base batch size of training/generated data' )

    parser.add_argument( '--num_main_iters', type = int, default = 300000, help = 'number of main iterations of training generator and discriminator' )
    parser.add_argument( '--num_disc_iters', type = int, default = 5, \
      help = "number of discriminator iterations in one main iteration through generator and discriminator;" + \
             " suggested to train to optimality so that minimizing generator's loss is the same as minimizing the JS-divergence for many types of loss functions" )

    parser.add_argument( '--lr_base', type = float, default = .0001, help = 'base (max) learning rate; can vary depending on dataset and model' )
    parser.add_argument( '--lr_sched', type = str.casefold, default = None, \
      choices = [ None, 'linear decay', 'custom' ], help = 'learning rate scheduler type' )

    parser.add_argument( '--beta2', type = float, default = .9, help = 'variance/RMSprop EWMA decay coefficient (default recommended for Adam)' )

    parser.add_argument( '--res_samples', type = int, default = 64, choices = [ 32, 64 ], help = 'resolution of generated samples' )
    parser.add_argument( '--res_dataset', type = int, default = 64, \
      help = "this is the size of your dataset's shorter dimension (H or W), before any resampling is done;" + \
             " this also places an upper limit on the sample resolution (i.e. the max value that res_samples above can be)" )

    parser.add_argument( '--blur_type', type = str.casefold, default = None, choices = [ None, 'binomial', 'gaussian', 'box' ], \
      help = 'low-pass filter blurring operation to apply after upsampling and before downsampling; these effectively change the upsampling/downsampling operation,' + \
             ' for e.g., setting `binomial` converts nearest neighbor upsampling into bilinear upsampling, setting `box` just average pools' )

    parser.add_argument( '--eps_drift', type = float, default = 0., help = 'drift loss multiplicative coefficient; 0. means no drift loss' )

    parser.add_argument( '--len_latent', type = int, default = 128, help = 'number of elements that compose latent vector z' )

    parser.add_argument( '--nonlinearity', type = str.casefold, default = 'relu', choices = [ 'leaky relu', 'relu', 'tanh' ], \
      help = 'which nonlinearity to use in generator and discriminator' )
    parser.add_argument( '--leakiness', type = float, default = .01, help = 'leakiness of Leaky ReLU if used (otherwise ignored)' )
    # parser.add_argument( '--use_tanh', type = bool, default = True, help = 'whether to use tanh at the end of the generator' )

    parser.add_argument( '--use_equalized_lr', type = str2bool, nargs = '?', const = True, default = False, \
      help = 'whether to use equalized learning rate in generator and discriminator' )

  # ............................................................................ #

  # Progressive GAN/StyleGAN Keyword Arguments:
  elif _model_type in ( 'ProGAN', 'StyleGAN', ):
    parser.add_argument( '--batch_size', type = int, default = BS, help = 'base batch size of training/generated data' )
    parser.add_argument( '--bs_dict', type = dict, default = { 4:BS, 8:BS, 16:BS, 32:BS, 64:BS, 128:BS, 256:BS, 512:BS//2, 1024:BS//4 }, \
      help = 'resolution-dependent batch sizes;' + \
             ' change these based on the memory-requirements of your device; higher resolutions than 1024 are also supported' )

    parser.add_argument( '--num_disc_iters', type = int, default = 1, \
      help = "number of discriminator iterations in one main iteration through generator and discriminator;" + \
             " no need to train discriminator to optimality here (for faster convergence)" )
    parser.add_argument( '--nimg_transition', type = int, default = NIMG_TRANSITION, \
      help = 'number of real images seen by discriminator before transitioning to next fade-in/stabilization phase' )

    parser.add_argument( '--lr_base', type = float, default = .001, help = 'base (max) learning rate; can vary depending on dataset and model' )
    parser.add_argument( '--lr_sched', type = str.casefold, default = 'resolution dependent', \
      choices = [ None, 'resolution dependent', 'linear decay', 'custom' ], help = 'learning rate scheduler type' )

    parser.add_argument( '--beta2', type = float, default = .99, help = 'variance/RMSprop EWMA decay coefficient (default recommended for Adam)' )

    parser.add_argument( '--res_samples', type = int, default = 1024, choices = np.logspace( 2, 16, num = 15, base = 2, dtype = np.uint64 ), \
      help = 'final resoluton of generated samples' )
    parser.add_argument( '--res_dataset', type = int, default = 1024, choices = np.logspace( 2, 16, num = 15, base = 2, dtype = np.uint64 ), \
      help = "this is the size of your dataset's shorter dimension (H or W), before any resampling is done;" + \
             " this also places an upper limit on the sample resolution (i.e. the max value that res_samples above can be)" )

    parser.add_argument( '--blur_type', type = str.casefold, default = 'binomial', choices = [ None, 'binomial', 'gaussian', 'box' ], \
      help = 'low-pass filter blurring operation to apply after upsampling and before downsampling; these effectively change the upsampling/downsampling operation,' + \
             ' for e.g., setting `binomial` converts nearest neighbor upsampling into bilinear upsampling, setting `box` just average pools' )
    parser.add_argument( '--bit_exact_resampling', type = str2bool, nargs = '?', const = True, default = False, \
      help = "whether to use PIL's resampling methods (if `True`) or torch's resampling functions (if `False`) for the real data's skip connection;" + \
             " the torch resampling functions are faster" )

    parser.add_argument( '--eps_drift', type = float, default = .001, help = 'drift loss multiplicative coefficient; 0. means no drift loss' )

    parser.add_argument( '--len_latent', type = int, default = 512, help = 'number of elements that compose latent vector z' )

    parser.add_argument( '--nonlinearity', type = str.casefold, default = 'leaky relu', choices = [ 'leaky relu', 'relu', 'tanh' ], \
      help = 'which nonlinearity to use in generator and discriminator' )
    parser.add_argument( '--leakiness', type = float, default = .2, help = 'leakiness of Leaky ReLU if used (otherwise ignored)' )
    # parser.add_argument( '--use_tanh', type = bool, default = False, help = 'whether to use tanh at the end of the generator' )

    parser.add_argument( '--use_equalized_lr', type = str2bool, nargs = '?', const = True, default = True, \
      help = 'whether to use equalized learning rate in generator and discriminator' )
    parser.add_argument( '--normalize_z', type = str2bool, nargs = '?', const = True, default = True, \
      help = "whether to use pixelwise feature vector normalization on the latent vector z (regardless of --use_pixelnorm's value)" )

    parser.add_argument( '--mbstd_group_size', type = int, default = 4, \
      help = 'group size for minibatch standard deviation layer in the discriminator to help increase variation in generated images;' + \
             ' setting this to -1 instead will not apply this layer at all' )

    parser.add_argument( '--use_ewma_gen', type = str2bool, nargs = '?', const = True, default = True, \
      help = 'whether to compute an additional time-averaged (EWMA) generator that often yields higher-quality results than the original generator' )

    # .......................................................................... #

    # Progressive GAN-specific Keyword Arguments:
    if _model_type is 'ProGAN':
      parser.add_argument( '--num_main_iters', type = int, default = ( NIMG_TRANSITION // BS ) * 20, \
        help = 'number of main iterations of training generator and discriminator' )

      parser.add_argument( '--lr_fctr_dict', type = dict, default = { 4:1., 8:1., 16:1., 32:1., 64:1., 128:1., 256:1., 512:1., 1024:1.5 }, \
        help = 'resolution-dependent multiplicative factor to multiply lr_base by' )

      parser.add_argument( '--init_res', type = int, default = 4, choices = np.logspace( 2, 16, num = 15, base = 2, dtype = np.uint64 ), \
        help = 'starting resolution of Progressive GAN' )

      parser.add_argument( '--use_pixelnorm', type = str2bool, nargs = '?', const = True, default = True, \
        help = 'whether to use pixelwise feature vector normalization in generator' )

    # .......................................................................... #

    # StyleGAN-specific Keyword Arguments:
    elif _model_type is 'StyleGAN':
      parser.add_argument( '--num_main_iters', type = int, default = ( NIMG_TRANSITION // BS ) * 18, \
        help = 'number of main iterations of training generator and discriminator' )

      parser.add_argument( '--lr_fctr_dict', type = dict, default = { 4:1., 8:1., 16:1., 32:1., 64:1., 128:1.5, 256:2., 512:3., 1024:3. }, \
        help = 'resolution-dependent multiplicative factor to multiply lr_base by' )

      parser.add_argument( '--init_res', type = int, default = 8, choices = np.logspace( 2, 16, num = 15, base = 2, dtype = np.uint64 ), \
        help = 'starting resolution of StyleGAN' )

      parser.add_argument( '--len_dlatent', type = int, default = 512, \
        help = 'number of elements that compose disentangled (intermediate) latent vector w' )
      parser.add_argument( '--mapping_num_fcs', type = int, default = 8, \
        help = 'number of fully-connected layers that make up the mapping network' )
      parser.add_argument( '--mapping_lrmul', type = int, default = .01, \
        help = 'multiplicative factor for the learning rate in the mapping network to reduce instability' )

      parser.add_argument( '--use_noise', type = str2bool, nargs = '?', const = True, default = True, \
        help = 'whether to apply noise inputs to generator in order to control stochastic variation in generated images' )
      parser.add_argument( '--use_pixelnorm', type = str2bool, nargs = '?', const = True, default = False, \
        help = 'whether to use pixelwise feature vector normalization in generator' )
      parser.add_argument( '--use_instancenorm', type = str2bool, nargs = '?', const = True, default = True, \
        help = 'whether to use instance normalization in generator (recommended for AdaIN)' )

      # TODO: Implement option to style-mix more than just 2 latent vectors for the mixing regularization
      parser.add_argument( '--pct_mixing_reg', type = float, default = .9, \
        help = 'percent of images generated from 2 latent vectors z instead of 1; 0. means no mixing regularization' )

      parser.add_argument( '--beta_trunc_trick', type = float, default = .995, \
        help = 'decay coefficient for computing EWMA of disentangled latent vector w during training,' + \
               ' which will be used for the truncation trick in disentangled latent space W during evaluation' )
      parser.add_argument( '--psi_trunc_trick', type = float, default = .7, \
        help = 'multiplicative coefficient in range [0,1) for the truncation trick in disentangled latent space W during evaluation;' + \
               ' the smaller the value, the more the truncation' )
      parser.add_argument( '--cutoff_trunc_trick', type = int, default = 4, \
        help = 'final generator block ( = log2(res) - 1 ) to use the computed psi-truncated disentangled latent vector w during evaluation' )

  # ---------------------------------------------------------------------------- #

  config = parser.parse_args( )

  data_config = get_current_configuration( 'data_config', raise_exception = False )

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
  # Post-processing:
  # ----------------

  config.dev = torch.device( config.dev )

  if config.pin_memory and config.dev != torch.device( 'cuda' ):
    config = None
    raise ValueError( '--pin_memory should be set to `False` if not using CUDA.' )

  if config.random_seed == -1:
    np.random.seed( None )
    if config.n_gpu > 1:
      torch.cuda.seed_all()  # multi-GPU, ignored if CPU
    else:
      torch.seed()  # single-processor, CUDA or CPU
  elif 0 <= config.random_seed < 2**32:
    np.random.seed( config.random_seed )
    if config.n_gpu > 1:
      torch.cuda.manual_seed_all( config.random_seed )  # multi-GPU, ignored if CPU
    else:
      torch.manual_seed( config.random_seed )  # single-processor, CUDA or CPU
    torch.backends.cudnn.deterministic = True
    if config.enable_cudnn_autotuner:
      config.enable_cudnn_autotuner = False
      print( 'To maintain determinism, the CuDNN auto-tuner has been disabled...' )
  else:
    config = None
    raise ValueError( "--random_seed must either be -1 for random seeding or" + \
                      " be in the range [0,2**32) to accommodate numpy's and" + \
                      " torch's seeding specifications." )

  torch.backends.cudnn.benchmark = config.enable_cudnn_autotuner

  if _model_type in ( 'ProGAN', 'StyleGAN', ):
    _bs = config.batch_size
    config.bs_dict = { 4:_bs, 8:_bs, 16:_bs, 32:_bs, 64:_bs, 128:_bs, 256:_bs, 512:_bs//2, 1024:_bs//4 }
    if config.mbstd_group_size < -1 or not config.mbstd_group_size:
      raise ValueError( "--mbstd_group_size must either be -1 to indicate not applying" + \
                        " minibatch standard deviation or a positive integer indicating" + \
                        " the group size for the minibatch standard deviation layer." )

  config.save_samples_dir.mkdir( parents = True, exist_ok = True )
  config.save_model_dir.mkdir( parents = True, exist_ok = True )

  configs_dir = str( os.path.abspath( os.path.dirname( __file__ ) ) )

  if data_config is not None:
    for n, transform in enumerate( data_config.ds_transforms.transforms ):
      _num_rsz = 0
      # NOTE: messy control flow below
      if ( isinstance( transform, partial ) and transform.func is transforms.Resize ) or \
        isinstance( transform, transforms.Resize ):
        _num_rsz += 1
        if _num_rsz < 2:
          if _model_type == 'ResNet GAN':
            _res = config.res_samples
          elif _model_type in ( 'ProGAN', 'StyleGAN', ):
            _res = config.init_res
          if isinstance( transform, partial ):
            data_config.ds_transforms.transforms[n] = \
              data_config.ds_transforms.transforms[n]( size = ( _res, _res, ) )
          elif data_config.ds_transforms.transforms[n].size[0] != _res:
            data_config.ds_transforms.transforms[n].size = ( _res, _res, )
        else:
          raise RuntimeWarning( 'Warning: More than 1 "Resize" transform found;' + \
                                ' only updated the first "Resize" transform.' )

    with open( configs_dir + '/.data_config.p', 'wb' ) as f:
      pickle.dump( data_config, f, protocol = 3 )

  config.model = _model_type

  # print( config )

  with open( str( Path.home()/'.configs_dir.txt' ), 'wb' ) as f:
    f.write( configs_dir.encode() )

  with open( configs_dir + '/.config.p', 'wb' ) as f:
    pickle.dump( config, f, protocol = 3 )