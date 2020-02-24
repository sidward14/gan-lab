# -*- coding: UTF-8 -*-

# TODO: Incorporate LMDB-based dataloading

"""Utilities for constructing the appropriate set of torch Datasets & DataLoaders.

Functions "prepare_dataset(...)" and "prepare_dataloader(...)" construct a set
of torch Datasets and DataLoaders respectively based off of configurations
specified in data_config.py and config.py; both of these modules have to be
run atleast once (see their respective docstrings for instructions on running
them) before "prepare_dataset(...)" and "prepare_dataloader(...)" can be run.

  Typical usage examples:

  train_ds, valid_ds = prepare_dataset( data_config )
  train_dl, valid_dl, z_valid_dl = prepare_dataloader( config, data_config, train_ds, valid_ds )

The output DataLoaders can then used as inputs into the train() method of the
appropriate Learner (which was determined when you ran config.py), as is done,
for example, when running train.py on the command-line.

The functions in this module can also be used externally as data-construction
utilities.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from .latent_utils import gen_rand_latent_vars, concat_rand_classes_to_z
from _int import get_current_configuration, \
                 TRAINING_SET_DIRNAME, VALID_SET_DIRNAME, TORCHVISION_CATERED_DATASETS
from data_config import get_dataset_img_extension

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets.folder import *
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# torch Dataset Utilities:
# ------------------------

def prepare_dataset( data_config ):
  """Construct torch Dataset(s) based on configuration specified when data_config.py was last ran."""
  dataset_maker = partial(
    make_torchvision_dataset,
    dataset = data_config.dataset,
    dataset_dir = data_config.dataset_dir,
    ds_transforms = data_config.ds_transforms
  )

  valid_ds = None
  if data_config.is_torchvision_catered_ds or data_config.ds_ext in IMG_EXTENSIONS:
    train_ds = dataset_maker( is_training_set = True )
    if data_config.include_valid_set:
      valid_ds = dataset_maker( is_training_set = False )
  else:  # use DatasetFolderSingleClass
    train_ds = dataset_maker( is_training_set = True,
                              loader = data_config.ds_loader, extensions = ( data_config.ds_ext, ) )
    if data_config.include_valid_set:
      valid_ds = dataset_maker( is_training_set = False,
                                loader = data_config.ds_loader, extensions = ( data_config.ds_ext, ) )

  return train_ds, valid_ds

# ............................................................................ #

# TODO: Implement class-conditioning option (Mirza & Osindero, 2014) for discriminator input.
def make_torchvision_dataset( dataset, dataset_dir, is_training_set = True, ds_transforms = None, *args, **kwargs ):
  if not isinstance( dataset_dir, ( str, Path, ) ):
    raise TypeError( '"dataset_dir" must be of type "str" or "pathlib.Path"' )
  dataset_dir = str( dataset_dir )

  dataset_title = dataset.casefold()
  # TODO: Add more torchvision-catered datasets:
  if dataset_title in TORCHVISION_CATERED_DATASETS['LSUN Bedrooms'][0]:
    config = get_current_configuration( 'config', raise_exception = False )
    if config is not None:
      if config.res_dataset > 256:
        message = f'WARNING: config.res_dataset currently set to {config.res_dataset},' + \
                  f' but recommended to set --res_dataset to 256 or below when using LSUN Bedrooms!'
        raise RuntimeWarning( message )
    classes_categories = TORCHVISION_CATERED_DATASETS['LSUN Bedrooms'][1]
    classes = classes_categories[0] if is_training_set else classes_categories[1]
    dataset = datasets.LSUN(
      root = dataset_dir, classes = classes, transform = ds_transforms
    )
  elif dataset_title in TORCHVISION_CATERED_DATASETS['CIFAR-10'][0]:
    config = get_current_configuration( 'config', raise_exception = False )
    if config is not None:
      if config.res_dataset > 32:
        message = f'WARNING: config.res_dataset currently set to {config.res_dataset},' + \
                  f' but recommended to set --res_dataset to 32 or below when using CIFAR-10!'
        raise RuntimeWarning( message )
    classes_categories = TORCHVISION_CATERED_DATASETS['CIFAR-10'][1]
    classes = classes_categories[0] if is_training_set else classes_categories[1]
    dataset = datasets.CIFAR10(
      root = dataset_dir, train = classes, transform = ds_transforms, download = True
    )
  else:
    # Custom Datasets:
    imgs_dirname = TRAINING_SET_DIRNAME if is_training_set else VALID_SET_DIRNAME
    if get_dataset_img_extension( dataset_dir + '/' + imgs_dirname ) in IMG_EXTENSIONS:
      dataset = ImageFolderSingleClass( root = dataset_dir, category = imgs_dirname, transform = ds_transforms )
    else:
      if 'loader' not in kwargs or kwargs[ 'loader' ] is None:
        raise ValueError( 'Generic DatasetFolderSingleClass requires input `loader:callable` auxiliary argument.' )
      raise RuntimeWarning( '\nUsing generic DatasetFolderSingleClass...\n' + \
                            ' WARNING: The input "loader" argument for this class that acts on your data must output an object that' + \
                                       ' can be converted into a PIL Image object (e.g. an object of type `np.uint8` or `torch.uint8`).\n' )
      loader = kwargs.pop( 'loader' )
      dataset = DatasetFolderSingleClass( root = dataset_dir, category = imgs_dirname,
                                          loader = loader, transform = ds_transforms, **kwargs )
  return dataset

class DatasetFolderSingleClass( datasets.DatasetFolder ):
  """torchvision's DatasetFolderClass but with the ability to just use 1 class."""
  def __init__( self, root, category, loader, extensions = None, transform = None,
                target_transform = None, is_valid_file = None ):
    super( datasets.DatasetFolder, self ).__init__( root, transform = transform,
                                                    target_transform = target_transform )
    # classes, class_to_idx = self._find_classes(self.root)
    classes = [ category ]; class_to_idx = { category: 0 }
    samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
    if len(samples) == 0:
      raise ( RuntimeError( 'Found 0 files in subfolders of: ' + self.root + '\n'
                            'Supported extensions are: ' + ','.join( extensions ) )
            )

    self.loader = loader
    self.extensions = extensions

    self.classes = classes
    self.class_to_idx = class_to_idx
    self.samples = samples
    self.targets = [s[1] for s in samples]

class ImageFolderSingleClass( DatasetFolderSingleClass ):
  """torchvision's ImageFolderClass but with the ability to just use 1 class."""
  def __init__( self, root, category, transform = None, target_transform=None,
                loader = default_loader, is_valid_file = None ):
    super( ImageFolderSingleClass, self ).__init__( root, category, loader,
                                                    IMG_EXTENSIONS if is_valid_file is None else None,
                                                    transform = transform,
                                                    target_transform = target_transform,
                                                    is_valid_file = is_valid_file
                                                  )
    self.imgs = self.samples

# ............................................................................ #

# TODO: Implement more types of loaders
def npy_loader( path ):
  return torch.from_numpy( np.load( path ) ).squeeze()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# torch DataLoader Utilities:
# ---------------------------

def prepare_dataloader( config, data_config, train_ds, valid_ds = None ):
  """Construct torch DataLoader(s) based on configuration specified when config.py and data_config.py was last ran."""
  if data_config.include_valid_set:
    # DataLoader for latent vectors for validation:
    z_valid = gen_rand_latent_vars(
                num_samples = len( valid_ds ),
                length = config.len_latent,
                distribution = config.latent_distribution,
                device = 'cpu'
              )
    if config.class_condition or config.use_auxiliary_classifier:
        z_labels = torch.randint( 0, config.num_classes, ( len( z_valid ), 1, ),
                                  dtype = torch.int64, device = 'cpu' )
    if config.class_condition:
        z_valid = concat_rand_classes_to_z( z = z_valid, num_classes = config.num_classes,
                                            z_labels = z_labels, device = 'cpu' )
    if config.class_condition or config.use_auxiliary_classifier:
        z_labels.squeeze_()
        z_valid_ds = TensorDataset( z_valid, z_labels )
    else:
        z_valid_ds = TensorDataset( z_valid )

  # DataLoader(s) for training data:
  dataloader = configure_dataloader_for_hardware( num_workers = config.num_workers, pin_memory = config.pin_memory )

  valid_dl = None; z_valid_dl = None
  if config.model == 'ResNet GAN':
    train_dl = dataloader( dataset = train_ds, batch_size = config.batch_size,
                           shuffle = True, drop_last = True )
    if data_config.include_valid_set:
      valid_dl = dataloader( dataset = valid_ds, batch_size = config.batch_size,
                             shuffle = False, drop_last = False )
      z_valid_dl = dataloader( dataset = z_valid_ds, batch_size = config.batch_size,
                               shuffle = False, drop_last = False )
  elif config.model in ( 'ProGAN', 'StyleGAN', ):
    train_batch_sampler = BatchSampler( sampler = RandomSampler( data_source = train_ds ),
                                        batch_size = config.bs_dict[ config.init_res ], drop_last = True )
    train_dl = dataloader( dataset = train_ds, batch_sampler = train_batch_sampler )
    if data_config.include_valid_set:
      valid_batch_sampler = BatchSampler( sampler = SequentialSampler( data_source = valid_ds ),
                                          batch_size = config.bs_dict[ config.init_res ], drop_last = False )
      valid_dl = dataloader( dataset = valid_ds, batch_sampler = valid_batch_sampler )
      z_valid_batch_sampler = BatchSampler( sampler = SequentialSampler( data_source = z_valid_ds ),
                                            batch_size = config.bs_dict[ config.init_res ], drop_last = False )
      z_valid_dl = dataloader( dataset = z_valid_ds, batch_sampler = z_valid_batch_sampler )
  else:
    message = f'Model type {config.model} not supported, thus could not construct torch DataLoader.'
    raise ValueError( message )

  return train_dl, valid_dl, z_valid_dl

# ............................................................................ #

def configure_dataloader_for_hardware( num_workers, pin_memory ):
  dataloader = partial(
    DataLoader,
    num_workers = num_workers,
    pin_memory = pin_memory
  )
  return dataloader

class DataLoaderEx( DataLoader ):
  """Customize your own DataLoader."""
  def __init__( self, dataset , bs ):
    super( DataLoaderEx, self ).__init__( dataset, bs )
    raise NotImplementedError( 'Class "DataLoaderEx" not yet implemented.' )