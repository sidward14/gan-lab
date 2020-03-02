# -*- coding: UTF-8 -*-

"""Configures package for the user-specified dataset. Contains useful utilities.

Run 'python data_config.py [dataset] [dataset_dir]' on the command-line to
configure (or re-configure) this package for the dataset you wish to use, and
optionally specify any set of '--[keyword argument]' to override any default
keyword arguments pertaining to image transforms or validation set creation.

  Typical usage examples:

  $ python data_config.py FFHQ path/to/datasets/ffhq  [OR]
  $ python data_config.py FFHQ path/to/datasets/ffhq --enable_mirror_augmentation  [OR]
  $ python data_config.py FFHQ path/to/datasets/ffhq --include_valid_set=False

After running this module (as well as running config.py), you will be able to
construct (at any later time) a set of torch Datasets and DataLoaders that is
specific to your input dataset configuration & hyperparameters by running
the "prepare_dataset(...)" and "prepare_dataloader(...)" functions respectively,
from the utils.data_utils.py module.

This module can also be imported externally and used for its various utilities.

If you would like to see a list of what each argument does,
run 'python data_config.py [dataset] [dataset_dir] -h' on the command-line.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

import sys
if '_int' in sys.modules or __name__ == '__main__':
  from _int import str2bool, get_current_configuration, \
                   TRAINING_SET_DIRNAME, VALID_SET_DIRNAME, \
                   TORCHVISION_CATERED_DATASETS
elif 'gan_lab._int' in sys.modules:
  from ._int import str2bool, get_current_configuration, \
                    TRAINING_SET_DIRNAME, VALID_SET_DIRNAME, \
                    TORCHVISION_CATERED_DATASETS

import os
import argparse
from typing import Union
from functools import partial
from pathlib import Path
import shutil
import pickle

import numpy as np
import torch
from torchvision.datasets.folder import *
from torchvision import transforms

from PIL import Image

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# TODO: Implement ability to transform without having to use PIL transforms/convert to PIL Image object
USE_PIL_TRANSFORMS = True

DEFAULT_STATS = { 'mean' : [ 0.5, 0.5, 0.5 ], 'std' : [ 0.5, 0.5, 0.5 ] }
LSUN_BEDROOM_STATS = { 'mean' : [ 0.5, 0.5, 0.5 ], 'std' : [ 0.5, 0.5, 0.5 ] }
CIFAR10_STATS = { 'mean' : [ .491, .482, .447 ], 'std' : [ .247, .243, .261 ] }
CELEBAHQ_STATS = { 'mean' : [ .5, .5, .5 ], 'std' : [ .5, .5, .5 ] }

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# ---------------------------------------------------------------------------- #

# Validation Set creation:
# ------------------------

def make_valid_set( dataset_dir:Path, training_set_dirname = 'train', valid_set_dirname = 'valid', valid_set_pct = None, valid_set_idxs:np.integer = None ):
  """Partitions dataset into training set and validation set based on either input % validation or validation indices of training set."""
  try:
    dataset_dir = Path( dataset_dir )
  except:
    raise TypeError( 'Input dataset path must be convertible to a "pathlib.Path" instance.' )

  assert dataset_dir.exists()

  if valid_set_pct is not None:
    assert isinstance( valid_set_pct, ( float, int, ) )
    assert ( 0 <= valid_set_pct <= 1 )
  _bool_valid_idx_np = False
  if valid_set_idxs is not None:
    assert isinstance( valid_set_idxs, np.integer )
    if valid_set_idxs.size > 0: _bool_valid_idx_np = True

  training_set_path = dataset_dir/training_set_dirname
  training_set_path_str = str( training_set_path )
  valid_set_path = dataset_dir/valid_set_dirname
  valid_set_path_str = str( valid_set_path )

  if valid_set_pct is not None or valid_set_idxs is not None:

    if valid_set_pct != _bool_valid_idx_np:  # xor

      if not ( training_set_path.exists() and training_set_path.is_dir() ):
        print( f'Creating training set at "{training_set_path_str}"...\n' )
        training_set_path.mkdir( exist_ok = False )
        for f in dataset_dir.iterdir():
          if f.is_file():
            shutil.move( str( f ), training_set_path_str )

      if valid_set_path.exists() and valid_set_path.is_dir():
        print( f'Validation set already exists in directory "{valid_set_path_str}".\n' + \
               f'Moving validation set data back to "{training_set_path_str}" before re-making validation set...\n' )
        for f in valid_set_path.iterdir():
          if f.is_file():
            shutil.move( str( f ), training_set_path_str )

      dataset_sz = len( [ f for f in training_set_path.iterdir() if f.is_file() ] )

      valid_set_path.mkdir( exist_ok = True )
      if valid_set_pct:
        valid_set_idxs = np.random.choice(
          dataset_sz, size = round( valid_set_pct * dataset_sz ), replace = False
        )
      print( f'{len(valid_set_idxs)} images will now be moved from "{training_set_path_str}" to "{valid_set_path_str}"...\n' )
      for n, f in enumerate( training_set_path.iterdir() ):
        if n in valid_set_idxs and f.is_file():
          shutil.move( str( f ), valid_set_path_str )

    elif valid_set_pct and _bool_valid_idx_np:
      raise ValueError( 'Cannot specify both a validation pct and array of indices. Please choose one.' )

  else:
    if not ( training_set_path.exists() and training_set_path.is_dir() ) or \
       not ( valid_set_path.exists() and valid_set_path.is_dir() ):
      raise ValueError( 'validation set desired but validation set cannot be created because' + \
                        ' no % validation or validation indices specified' )
    raise RuntimeWarning( 'No % validation or validation indices specified.' + \
                          ' Therefore nothing has been done with regards to validation set creation (or re-creation).\n' )

  imgs_dir = training_set_path

  return imgs_dir

# ---------------------------------------------------------------------------- #

# Data Normalization:
# -------------------

# NOTE: Input stats_type = 'default' into the function below to map pixel intensity
#       value 0 to -1 and pixel intensity value 255 to 1.
def normalize_center_standardize( dataset = 'default', normalize = False, standardize = False ):
  """Normalize, center, and/or standardize data to [-1,1].
     These assume that your stats (constants listed above) are for the normalized (i.e. range [0,1]) versions of these images.
  """
  dataset = dataset.casefold()

  if standardize:
    if dataset == 'default':
      stats = DEFAULT_STATS
    elif dataset in ( 'lsun bedrooms', 'lsun-bedrooms', 'lsunbedrooms', ):
      stats = LSUN_BEDROOM_STATS
    elif dataset in ( 'cifar-10', 'cifar10', 'cifar 10', ):
      stats = DEFAULT_STATS  # CIFAR10_STATS
    elif dataset in ( 'celeba-hq', 'celebahq', 'celeba hq', ):
      stats = CELEBAHQ_STATS
    # elif dataset == 'ffhq':
    #   raise NotImplementedError( 'FFHQ dataset configuration support not yet implemented.' )
    else:
      raise NotImplementedError(
        "input dataset's statistics not yet implemented. Please use dataset = `'default'`" + \
        " instead for default (and recommended) standardization, or equivalently just set standardization = `False`."
      )

  _nfctr = 1.
  if normalize:
    _nfctr = 255.

  if standardize:
    mean = [ m*_nfctr for m in stats[ 'mean' ] ]
    std = [ s*_nfctr for s in stats[ 'std' ] ]
  else:
    mean = [ m*_nfctr for m in DEFAULT_STATS[ 'mean' ] ]
    std = [ s*_nfctr for s in DEFAULT_STATS[ 'std' ] ]

  return mean, std

def normalize_notorch( dataset ):
  """Normalize data when torch is unavailable."""
  raise NotImplementedError( '"normalize_notorch( data )" not yet implemented.' )

# ---------------------------------------------------------------------------- #

# Misc. Utilities:
# ----------------

def get_dataset_img_extension( imgs_dir ):
  """Quick (non-comprehensive) function to get the extension of the dataset's data."""
  fs = Path( imgs_dir ).iterdir()
  f0 = next( fs )
  while not f0.is_file():
    f0 = next( fs )
  fs = iter( reversed( list( Path( imgs_dir ).iterdir() ) ) )
  ff = next( fs )
  while not ff.is_file():
    ff = next( fs )
  if f0.suffix.casefold() == ff.suffix.casefold():
    return f0.suffix.casefold()
  else:
    raise Exception( "Unable to identify the extension of the dataset's data" )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Set up and configure your input dataset:
# ----------------------------------------

if __name__ == '__main__':

  parser = argparse.ArgumentParser( prog = 'GAN Lab', description = 'StyleGAN, ProGAN, and ResNet GANs (dataset configuration)' )

  if len( sys.argv ) < 3:
    raise TypeError( 'data_config.py requires atleast 2 arguments (the dataset name and dataset directory), and then optional keyword arguments (prefixed with "--").\n' + \
                     ' Example: "$ python data_config.py FFHQ path/to/datasets/ffhq"' )

  # .......................................................................... #

  # POSITIONAL (REQUIRED) ARGUMENTS:
  # --------------------------------

  parser.add_argument( 'dataset', type = str.casefold, help = 'input dataset name (e.g. "FFHQ") used to train the model' )
  parser.add_argument( 'dataset_dir', type = Path, help = 'root directory where training data is stored' )

  # .......................................................................... #

  # KEYWORD (OPTIONAL) ARGUMENTS:
  # -----------------------------

  # Image Data Transforms
  # transforms performed in the order listed below; they are not implemented if not specified:
  parser.add_argument( '--center_crop_dim', type = int, help = 'square size of image after center cropping' )
  parser.add_argument( '--random_crop_dim', type = int, help = 'square size of image after random cropping' )
  parser.add_argument( '--dataset_downsample_type', type = str.casefold, default = 'average', choices = [ 'nearest', 'average', 'box', 'bilinear' ], \
    help = 'type of downsampling to conduct on dataset to get it to the target resolution for the model; for ResNet GANs, this target resolution is config.res_samples;' + \
           ' for ProGAN and StyleGAN, this target resolution is every resolution stage from config.init_res to the final resolution (i.e. config.res_samples) inclusive;' + \
           ' (side-note: if config has not yet been initialized, the ds_transforms attribute of data_config will show Resize as a "functools.partial" function until config is initialized)' )
  parser.add_argument( '--enable_mirror_augmentation', type = str2bool, nargs = '?', const = True, default = False, \
    help = 'whether to enable mirror augmentation (i.e. horizontal-flip augmentation)' )
  parser.add_argument( '--standardize', type = str2bool, nargs = '?', const = True, default = False, \
    help = 'optional (not recommended) standardization of pixels based on dataset statistics;' + \
           ' otherwise, map lowest possible pixel value to -1. and highest possible pixel value to 1 by default.' )

  # Validation Set Creation:
  parser.add_argument( '--include_valid_set', type = str2bool, nargs = '?', const = True, default = True, \
    help = "whether the input dataset should include a validation set during training (and therefore create a validation DataLoader);" + \
           " this flag also signals to create (or re-create) a validation set based on the 2 arguments below;" + \
           " however, if the dataset already comes pre-installed with a validation set (like with many torchvision-catered" + \
           " datasets such as LSUN Bedrooms or CIFAR-10), then all aspects considering validation set creation (or re-creation) are ignored," + \
           " and the 2 arguments below mean nothing (but the flag can still be used to signal that you want to train with the pre-installed validation set)" )
  parser.add_argument( '--valid_set_pct', type = float, default = .01, \
    help = "percent of training set to convert into validation set; ignored if --include_valid_set (above) == `False`;" + \
           " else if validation set already exists in dataset_dir (top), then this replaces the original validation set with this" + \
           " new one and moves the original validation set data back into the training set (you do not want to do this if goal is reproducibility," +\
           " in which case, set this equal to `None` (and keep the below argument equal to `None` as well) and nothing will happen)" )
  parser.add_argument( '--valid_set_idxs', type = np.integer, default = None, \
    help = "set of indices in the training set that will be used to create validation set; ignored if --include_valid_set (2 above) == `False`;" + \
           " else if validation set already exists in dataset_dir (top), then this replaces the original validation set with this" + \
           " new one and moves the original validation set data back into the training set (you do not want to use this if goal is reproducibility);" + \
           " overwrites --valid_set_pct (above)" )

  parser.add_argument( '--ds_loader', type = str.casefold, \
    help = "name of any possible loader (a function) one wants to use to load a sample given its path (see utils.data_utils.py for built-in loader(s));" + \
           " only necessary if using the generic DatasetFolderSingleClass class for your torch Dataset (see utils.data_utils.py);" + \
           " typically, as long as the data's extension is a generic image extension or if using one of torchvision's curated dataset classes," + \
           " DatasetFolderSingleClass will not be instantiated and a loader will thus not be necessary" )

  # .......................................................................... #

  data_config = parser.parse_args( )

  config = get_current_configuration( 'config', raise_exception = False )

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
  # Post-processing:
  # ----------------

  assert data_config.dataset_dir.exists()


  data_config.is_torchvision_catered_ds = \
    any( [ data_config.dataset in ds[0] for ds in TORCHVISION_CATERED_DATASETS.values() ] )


  # Optional Validation Set Creation based on user input:
  imgs_dir = data_config.dataset_dir
  if not data_config.is_torchvision_catered_ds:
    if ( data_config.dataset_dir/TRAINING_SET_DIRNAME ).exists() and \
       ( data_config.dataset_dir/TRAINING_SET_DIRNAME ).is_dir():
      imgs_dir = data_config.dataset_dir/TRAINING_SET_DIRNAME
    if data_config.include_valid_set:
      imgs_dir = make_valid_set(
                   dataset_dir = data_config.dataset_dir,
                   training_set_dirname = TRAINING_SET_DIRNAME,
                   valid_set_dirname = VALID_SET_DIRNAME,
                   valid_set_pct = data_config.valid_set_pct,
                   valid_set_idxs = data_config.valid_set_idxs
                 )


  # Configure Image Data with user-specified Transforms:
  if USE_PIL_TRANSFORMS:
    if data_config.dataset_downsample_type is 'nearest':
      data_config.dataset_downsample_type = Image.NEAREST
    elif data_config.dataset_downsample_type in ( 'average', 'box', ):
      data_config.dataset_downsample_type = Image.BOX
    elif data_config.dataset_downsample_type is 'bilinear':
      data_config.dataset_downsample_type = Image.BILINEAR

    ds_transforms = [ ]
    if not data_config.is_torchvision_catered_ds:
      ds_ext = get_dataset_img_extension( imgs_dir )
      if ds_ext not in IMG_EXTENSIONS:  # if `True`, then not automatically converted to PIL Image
        ds_transforms.append( transforms.ToPILImage( ) )
      data_config.ds_ext = ds_ext
    if data_config.center_crop_dim is not None:
      ds_transforms.append( transforms.CenterCrop( data_config.center_crop_dim ) )
    if data_config.random_crop_dim is not None:
      ds_transforms.append( transforms.RandomCrop( data_config.random_crop_dim ) )
    if config is not None:
      if config.model == 'ResNet GAN':
        ds_transforms.append( transforms.Resize( size = ( config.res_samples, config.res_samples, ),
                                                 interpolation = data_config.dataset_downsample_type ) )
      elif config.model in ( 'ProGAN', 'StyleGAN', ):
        ds_transforms.append( transforms.Resize( size = ( config.init_res, config.init_res, ),
                                                 interpolation = data_config.dataset_downsample_type ) )
    else:
      ds_transforms.append(
        partial( transforms.Resize, interpolation = data_config.dataset_downsample_type )
      )
    if data_config.enable_mirror_augmentation:  # data is at target resolution after now
      ds_transforms.append( transforms.RandomHorizontalFlip( p = 0.5 ) )
    ds_transforms.append( transforms.ToTensor( ) )  # this normalizes: maps pixel 0 -> 0. and pixel 255 -> 1.
    mean, std = normalize_center_standardize(
                  dataset = data_config.dataset,
                  normalize = False,  # transforms.ToTensor( ) above already takes care of normalization
                  standardize = data_config.standardize
                )
    ds_transforms.append( transforms.Normalize( mean = mean, std = std ) )
    data_config.ds_transforms = transforms.Compose( ds_transforms )
    data_config.ds_mean = mean
    data_config.ds_std = std
  else:
    raise NotImplementedError( 'Transforms on training data that do not use PIL are not yet implemented.' )

  # print( data_config )

  configs_dir = str( os.path.abspath( os.path.dirname( __file__ ) ) )

  with open( str( Path.home()/'.configs_dir.txt' ), 'wb' ) as f:
    f.write( configs_dir.encode() )

  with open( configs_dir + '/.data_config.p', 'wb' ) as f:
    pickle.dump( data_config, f, protocol = 3 )