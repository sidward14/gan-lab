# -*- coding: UTF-8 -*-

import os.path
from setuptools import setup

def readme( ):

  with open( os.path.abspath(
    os.path.join(
      os.path.dirname( __file__ ),
      'README.md' ) ) ) as f:

    return f.read( )

setup(
  name = 'gan-lab',
  version = '0.4.2',
  author = 'Sidhartha Parhi',
  author_email = 'sidhartha.parhi@gmail.com',
  description = "StyleGAN, ProGAN, and ResNet GANs to experiment with",
  long_description = readme( ),
  long_description_content_type = 'text/markdown',
  keywords = 'GAN GAN-Zoo ML generative neural model',
  url = "https://github.com/sidward14/gan-lab",
  packages = [
    'gan_lab',
    'gan_lab.utils',
    'gan_lab.resnetgan',
    'gan_lab.progan',
    'gan_lab.stylegan',
  ],
  dependency_links = [ ],
  install_requires = [
    'numpy >= 1.17.2',
    'scipy',
    'pillow >= 6.2.0',
    'matplotlib >= 3.1.1',
    'indexed >= 1.1.0',
    'torch >= 1.3.0',
    'torchvision >= 0.4.1',
    'lmdb >= 0.97',
    'tqdm',
  ],
  python_requires = '>= 3.6',
  include_package_data = True,
  zip_safe = False
)
