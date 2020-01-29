# -*- coding: UTF-8 -*-

"""Backpropagation utility functions.
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from _int import FMAP_SAMPLES, SUPPORTED_ARCHS

from functools import partial

import torch
import torch.nn.functional as F

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Loss Functions:
# ---------------

def wasserstein_distance_gen( outb ):
  # return outb.mean().abs()
  return -( outb.mean() )

def nonsaturating_loss_gen( outb ):
  return F.binary_cross_entropy_with_logits( input = outb,
                                             target = torch.ones( *( outb.shape ) ).to( outb.device ),
                                             reduction = 'mean' )

def minimax_loss_gen( outb ):
  return -F.binary_cross_entropy_with_logits( input = outb,
                                              target = torch.zeros( *( outb.shape ) ).to( outb.device ),
                                              reduction = 'mean' )

def wasserstein_distance_disc( outb, yb ):
  """Assumes you sample a pair of real & fake each time."""
  # return -( ( outb - yb ).mean().abs() )
  return ( outb - yb ).mean()

def minimax_loss_disc( outb, yb ):
  return F.binary_cross_entropy_with_logits( input = outb,
                                             target = torch.zeros( *( outb.shape ) ).to( outb.device ),
                                             reduction = 'mean' ) + \
         F.binary_cross_entropy_with_logits( input = yb,
                                             target = torch.ones( *( outb.shape ) ).to( outb.device ),
                                             reduction = 'mean' )

def wasserstein_distance_with_gp_disc( outb, yb, nn_disc, gp_type = 'wgan-gp', lda = 10., gamma = 1. ):
  """Assumes you sample a pair of real & fake each time."""
  # return -( ( outb - yb ).mean().abs() ) + calc_gp( gp_type, nn_disc, outb, yb, lda, gamma )
  return ( outb - yb ).mean() + calc_gp( gp_type, nn_disc, outb, yb, lda, gamma )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Gradient Regularizers:
# ----------------------

def calc_gp( gp_type, nn_disc, gen_data, real_data, lda = 10., gamma = 1. ):
  """Gradient penalty with custom lambda and gamma values."""
  gp_type = gp_type.casefold()

  if gp_type in ( 'wgan-gp', 'r1', ):
    real_data = real_data.view(
      -1, FMAP_SAMPLES, real_data.shape[2], real_data.shape[3]
    )
  if gp_type in ( 'wgan-gp', 'r2', ):
    gen_data = gen_data.view(
      -1, FMAP_SAMPLES, gen_data.shape[2], gen_data.shape[3]
    )

  # whether to penalize the discriminator gradients on the real distribution, fake distribution, or an interpolation of both
  # TODO: Implement class-conditioning (in the discriminator) option (Mirza & Osindero, 2014).
  if gp_type == 'wgan-gp':
    eps = torch.rand( gen_data.shape[0], 1, 1, 1, device = gen_data.device )
    xb = ( eps * gen_data.detach() + \
         ( 1 - eps ) * real_data.detach() ).to( gen_data.device )
  elif gp_type == 'r1':
    xb = real_data.detach().to( real_data.device )
  elif gp_type == 'r2':
    xb = gen_data.detach().to( gen_data.device )

  # now start tracking in the computation graph again
  xb.requires_grad_( True )

  if nn_disc.__class__.__name__ in SUPPORTED_ARCHS[ 'AC Discriminators' ]:
    outb, _ = nn_disc( xb )
  elif nn_disc.__class__.__name__ in SUPPORTED_ARCHS[ 'Discriminators' ]:
    outb = nn_disc( xb )
  # print( xb.requires_grad, outb.requires_grad )
  outb_grads = torch.autograd.grad(
    outb,
    xb,
    grad_outputs = torch.ones( gen_data.shape[0] ).to( gen_data.device ),
    create_graph = True, retain_graph = True, only_inputs = True )[0]

  if gp_type == 'wgan-gp':
    if gamma != 1.:
      outb_gp = \
        ( ( outb_grads.norm( 2, dim = 1 ) - gamma )**2 / gamma**2 ).mean() * lda
    else:
      outb_gp = \
        ( ( outb_grads.norm( 2, dim = 1 ) - 1. )**2 ).mean() * lda
  elif gp_type in ( 'r1', 'r2', ):
    outb_gp = ( outb_grads.norm( 2, dim = 1 )**2 ).mean() * lda / 2.

  return outb_gp

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Optimizer:
# ----------

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