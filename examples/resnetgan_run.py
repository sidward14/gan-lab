# -*- coding: UTF-8 -*-

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from gan_zoo import get_current_configuration
from gan_zoo.utils.data_utils import prepare_dataset, prepare_dataloader
from gan_zoo.resnetgan.learner import GANLearner

# get most recent configurations:
config = get_current_configuration( 'config' )
data_config = get_current_configuration( 'data_config' )

# get DataLoader(s)
train_ds, valid_ds = prepare_dataset( data_config )
train_dl, valid_dl, z_valid_dl = prepare_dataloader( config, data_config, train_ds, valid_ds )

# instantiate GANLearner and train:
learner = GANLearner( config )
print( learner.gen_model.__class__.__name__, learner.disc_model.__class__.__name__ )
learner.train( train_dl, valid_dl, z_valid_dl )   # train for config.num_main_iters iterations
learner.config.num_main_iters = 300000
learner.train( train_dl, valid_dl, z_valid_dl )   # train for another 300000 iterations
