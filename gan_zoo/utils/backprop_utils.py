# -*- coding: UTF-8 -*-

"""Backpropagation utility functions.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from _int import FMAP_SAMPLES, SUPPORTED_ARCHS

from functools import partial

import torch

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def wasserstein_distance_gen( outb ):
  # return outb.mean().abs()
  return -( outb.mean() )

def wasserstein_distance_disc( outb, yb ):
  """Assumes you sample a pair of real & fake each time."""
  # return -( ( outb - yb ).mean().abs() )
  return ( outb - yb ).mean()

def wasserstein_distance_with_gp_disc( outb, yb, nn_disc, lda = 10., gamma = 1. ):
  """Assumes you sample a pair of real & fake each time."""
  # return -( ( outb - yb ).mean().abs() ) + calc_gp( nn_disc, outb, yb )
  return ( outb - yb ).mean() + calc_gp( nn_disc, outb, yb, lda, gamma )

def calc_gp( nn_disc, gen_data, real_data, lda = 10., gamma = 1. ):
  """Gradient penalty with custom lambda and gamma values."""
  eps = torch.rand( gen_data.shape[0], 1, device = gen_data.device )

  gen_data = gen_data.view(
    -1, FMAP_SAMPLES * gen_data.shape[2] * gen_data.shape[3]
  )
  real_data = real_data.view(
    -1, FMAP_SAMPLES * real_data.shape[2] * real_data.shape[3]
  )

  # linear interpolation
  # TODO: Implement class-conditioning (in the discriminator) option (Mirza & Osindero, 2014) for the interpolate.
  xb_interp = ( eps * gen_data.detach() + \
                ( 1 - eps ) * real_data.detach() ).to( gen_data.device )

  # now start tracking in the computation graph again
  xb_interp.requires_grad_( True )

  if nn_disc.__class__.__name__ in SUPPORTED_ARCHS[ 'AC Discriminators' ]:
    outb_interp, _ = nn_disc( xb_interp )
  elif nn_disc.__class__.__name__ in SUPPORTED_ARCHS[ 'Discriminators' ]:
    outb_interp = nn_disc( xb_interp )
  # print( xb_interp.requires_grad, outb_interp.requires_grad )
  outb_interp_grads = torch.autograd.grad(
    outb_interp,
    xb_interp,
    grad_outputs = torch.ones( gen_data.shape[0] ).to( gen_data.device ),
    create_graph = True, retain_graph = True, only_inputs = True )[0]

  if gamma != 1.:
    outb_interp_gp = \
      ( ( outb_interp_grads.norm( 2, dim = 1 ) - gamma )**2 / gamma**2 ).mean() * lda
  else:
    outb_interp_gp = \
      ( ( outb_interp_grads.norm( 2, dim = 1 ) - 1. )**2 ).mean() * lda

  return outb_interp_gp

def configure_adam_for_gan( lr_base, betas:tuple, eps = 1.e-8, wd = 0 ):
  assert isinstance( betas, tuple )

  adam_gan = partial(
    torch.optim.Adam,
    lr = lr_base,
    betas = betas,
    eps = eps,
    weight_decay = wd
  )

  return adam_gan