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
  name = 'gan-zoo',
  version = '0.0.5',
  author = 'Sidhartha Parhi',
  author_email = 'sidhartha.parhi@gmail.com',
  description = "StyleGAN, ProGAN, and ResNet GANs with an intuitive API and helpful features",
  long_description = readme( ),
  long_description_content_type = 'text/markdown',
  url = "https://github.com/sidward14/gan-zoo",
  packages = [
    'gan_zoo',
    'gan_zoo.utils',
    'gan_zoo.resnetgan',
    'gan_zoo.progan',
    'gan_zoo.stylegan',
  ],
  dependency_links = [ ],
  install_requires = [
    'numpy >= 1.17.2',
    'pillow >= 6.2.0',
    'matplotlib >= 3.1.1',
    'indexed.py >= 0.0.1',
    'torch >= 1.3.0',
    'torchvision >= 0.4.1',
  ],
  python_requires = '>= 3.6',
  include_package_data = True,
  zip_safe = False
)
