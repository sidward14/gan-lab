# -*- coding: UTF-8 -*-

"""Bare-bones implementation for instantiating and training with a Learner.

Constructs and trains a Learner of the appropriate model type, hyperparameters,
and dataset (all determined from running config.py and data_config.py prior
to this).

  Simply run:

  $ python train.py

Note that one can always just write their own script (or Jupyter Notebook)
where they instantiate a Learner and then train/evaluate with it
(see README.md for an example of doing this).
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from _int import get_current_configuration
from utils.data_utils import prepare_dataset, prepare_dataloader
from resnetgan.learner import GANLearner
from progan.learner import ProGANLearner
from stylegan.learner import StyleGANLearner

from pathlib import Path

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

SAVE_MODEL_PATH = './models/gan_model.tar'

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

if __name__ == '__main__':

  config = get_current_configuration( 'config' )
  data_config = get_current_configuration( 'data_config' )

  # Construct DataLoader(s) according to config and data_config:
  # ------------------------------------------------------------
  train_ds, valid_ds = prepare_dataset( data_config )
  train_dl, valid_dl, z_valid_dl = \
    prepare_dataloader( config, data_config, train_ds, valid_ds )

  # Instantiate GAN Learner:
  # ------------------------
  if config.model == 'ResNet GAN':
    learner = GANLearner( config )
  elif config.model == 'ProGAN':
    learner = ProGANLearner( config )
  elif config.model == 'StyleGAN':
    learner = StyleGANLearner( config )
  else:
    raise ValueError( 'Invalid config.model. The GAN Lab currently only' + \
                      ' supports ResNet GAN, Progressive GAN, or StyleGAN.' )

  # Train for config.num_main_iters iterations:
  # -------------------------------------------
  learner.train( train_dl, valid_dl, z_valid_dl )

  # Save Model:
  # -----------
  learner.save_model( SAVE_MODEL_PATH )