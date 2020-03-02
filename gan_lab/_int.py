# -*- coding: UTF-8 -*-

"""Used internally.
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import os
import argparse
import copy
from pathlib import Path
import pickle

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Currently Supported Learners:
SUPPORTED_LEARNERS = ( 'GANLearner', 'ProGANLearner', 'StyleGANLearner', )

# Currently Supported Architectures:
SUPPORTED_ARCHS = {
  'Generators'        : ( 'Generator32PixResnet', 'Generator64PixResnet',
                          'ProGenerator', 'StyleGenerator',
  ),
  'Discriminators'    : ( 'Discriminator32PixResnet', 'Discriminator64PixResnet',
                          'ProDiscriminator', 'StyleDiscriminator',
  ),
  'AC Discriminators' : ( 'DiscriminatorAC32PixResnet', 'DiscriminatorAC64PixResnet', )
}

# For custom datasts only:
TRAINING_SET_DIRNAME = 'train'
VALID_SET_DIRNAME = 'valid'

# For torchvision-catered datasets only:
TORCHVISION_CATERED_DATASETS = {
  'LSUN Bedrooms' : (
                       ( 'lsun bedrooms', 'lsun-bedrooms', 'lsunbedrooms', ),
                       ( [ 'bedroom_train' ], [ 'bedroom_val' ], ),
  ),
  'CIFAR-10'      : (
                       ( 'cifar-10', 'cifar10', 'cifar 10', ),
                       ( True, False, ),
  )
}

FMAP_SAMPLES = 3
RES_INIT = 4

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class NotConfiguredError( Exception ):
  """Raised when either config.py or data_config.py has not been run."""
  pass

def get_current_configuration( cfg, raise_exception = True ):

  try:
    with open( str( Path.home()/'.configs_dir.txt' ), 'rb' ) as f:
      configs_dir = f.readline().decode( 'utf8' ).strip()
  except FileNotFoundError:
    if raise_exception:
      message = f'Please run config.py or data_config.py atleast once before running this function.'
      raise NotConfiguredError( message )
    else:
      return None

  if cfg is 'config':
    _pickled_file = configs_dir + '/.config.p'
    module = 'config.py'
  elif cfg is 'data_config':
    _pickled_file = configs_dir + '/.data_config.p'
    module = 'data_config.py'
  else:
    raise ValueError( "Input configuration does not exist. Options are 'config' and 'data_config'." )

  try:
    with open( _pickled_file, 'rb' ) as f:
      return pickle.load( f )
  except FileNotFoundError:
    if raise_exception:
      message = f'Please run {module} atleast once in order to obtain your desired configuration.'
      raise NotConfiguredError( message )
    else:
      return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool( v ):
  """Enables intuitive boolean keyword argument specifications with argparse."""
  if isinstance( v, bool ):
    return v
  if v.casefold() in ( 'yes', 'true', 't', 'y', '1' ):
    return True
  elif v.casefold() in ( 'no', 'false', 'f', 'n', '0' ):
    return False
  else:
    raise argparse.ArgumentTypeError( 'Boolean value expected.' )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class LearnerConfigCopy( object ):
  def __init__( self, config, learner_class:str,
                nonredefinable_attrs:tuple, redefinable_from_learner_attrs:tuple ):
    assert ( isinstance( config, argparse.Namespace ) and 'model' in config.__dict__ )
    super( LearnerConfigCopy, self ).__init__()

    object.__setattr__( self, '__dict__', copy.deepcopy( config.__dict__ ) )

    if learner_class is 'GANLearner':
      self.__dict__[ 'model_name' ] = 'resnetgan'
    elif learner_class is 'ProGANLearner':
      self.__dict__[ 'model_name' ] = 'progan'
    elif learner_class is 'StyleGANLearner':
      self.__dict__[ 'model_name' ] = 'stylegan'
    else:
      message = f"Input learner_class argument set equal to {learner_class}." + \
                f" But currently, learner_class can only be\n" + \
                f"one of: [ '" + "', '".join( SUPPORTED_LEARNERS ) + "' ]"
      raise ValueError( message )

    self.__dict__[ 'learner_class' ] = learner_class
    self.__dict__[ '_nonredefinable_attrs' ] = nonredefinable_attrs
    self.__dict__[ '_redefinable_from_learner_attrs' ] = redefinable_from_learner_attrs

  def __setattr__( self, name, value ):
    if name in self._nonredefinable_attrs:
      message0 = f"{self.learner_class}().config.{name} attribute cannot be changed once {self.learner_class} is instantiated.\n"
      if name == 'model':
        message1 = f"Instead, please run 'python config.py {value}' on the command-line and then instantiate a new {self.learner_class}."
      else:
        message1 = f"Instead, please run 'python config.py {self.model_name} --{name}={value}' on the command-line and then instantiate a new {self.learner_class}."
      message = message0 + message1
      raise AttributeError( message )
    elif name in self._redefinable_from_learner_attrs:
      message = f"{self.learner_class}().config.{name} attribute cannot be changed.\n" + \
                f" Instead, please change {self.learner_class}().{name} to implement this change in the {self.learner_class} instance,\n" + \
                f" while {self.learner_class}().config.{name} will remain equal to its value when the {self.learner_class} was first initialized."
      raise AttributeError( message )
    else:
      super( LearnerConfigCopy, self ).__setattr__( name, value )

  def __str__( self ):
    print_obj = ''
    for k, v in vars( self ).items():
      if k not in ( '_nonredefinable_attrs', '_redefinable_from_learner_attrs', 'model_name', 'learner_class', ):
        print_obj += f'  {k}: {v}\n'
    return print_obj[:-1]