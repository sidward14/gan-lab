# -*- coding: UTF-8 -*-

"""Latent Space-specific utilities.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

import torch

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def gen_rand_latent_vars( num_samples, length, distribution = 'normal',
                          device = 'cuda' if torch.cuda.is_available() else 'cpu' ):
  if distribution == 'normal':
    z = torch.randn( num_samples, length, dtype = torch.float32, device = device )
  elif distribution == 'uniform':
    z = torch.rand( num_samples, length, dtype = torch.float32, device = device )

  return z

def concat_rand_classes_to_z( z, num_classes, z_labels = None, \
                              device = 'cuda' if torch.cuda.is_available() else 'cpu' ):
  assert z_labels.shape == ( len( z ), 1, )

  if z_labels is None:
    z_labels = torch.randint( 0, num_classes, ( len( z ), 1, ), dtype = torch.int64, device = device )

  labels_one_hot = torch.zeros( len( z ), num_classes, dtype = torch.float32, device = device )
  labels_one_hot.scatter_( 1, z_labels, 1 )

  return torch.cat( ( z, labels_one_hot ), dim = 1 )