# GAN Zoo

<p align="left">
<img align="center" src ="https://github.com/sidward14/gan-zoo/blob/master/examples/gif/106.png" height="432" width="432"/>
<img align="center" src ="https://github.com/sidward14/gan-zoo/blob/master/examples/gif/672.png" height="432" width="432"/>
</p>

__Currently supports:__
+ StyleGAN
+ ProGAN
+ ResNet GANs

Each GAN model's default settings exactly emulates its respective most recent official implementation, but at the same time this package features a simple interface (config.py) where the user can quickly tune an extensive list of hyperparameter settings to their choosing.

Comes with additional features such as supervised learning capabilities, an easy-to-use interface for saving/loading the model state, flexible learning rate scheduling (and re-scheduling) capabilities, and more.

This package aims for an intuitive API without sacrificing any complexity anywhere.

--------------------------------------------------------------------------------

In your virtual environment (e.g. a conda virtual environment), run:
  ~~~
  $ pip install gan_zoo
  ~~~
This will install all necessary dependencies for you and will enable the option to use the package like an API (see "Jupyter Notebook (or Custom Script) Usage" below).

## Basic Usage on Command-line

__Clone this repo__, then simply running the following to configure your model & dataset and train your chosen model:
  ~~~
  $ python config.py [model] [--optional_kwargs]
  $ python data_config.py [dataset] [dataset_dir] [--optional_kwargs]
  $ python train.py
  ~~~
If you would like to see a list of what each argument does, run '$ python config.py [model] -h' or '$ python data_config.py [dataset] [dataset_dir] -h' on the command-line.

### ProGAN Example:

A ProGAN Generator that yields 128x128 images _(higher resolutions coming soon)_ like the ones below can be created by running the following 3 lines:
  ~~~
  $ python config.py progan --res_samples=128 --num_main_iters=1050000 --batch_size=8
  $ python data_config.py CelebA-HQ path/to/datasets/celeba_hq --enable_mirror_augmentation
  $ python train.py
  ~~~

  <p align="center">
  <img align="center" src ="https://github.com/sidward14/gan-zoo/blob/master/examples/gif/image_grids.gif"/>
  </p>
  <br>

By default (see config.py), the image grid above is saved periodically during training into the working directory of config.py (into the ./gan_zoo/samples folder) every 1000 iterations.

### StyleGAN Example:

A StyleGAN Generator can be created by running the following 3 lines (for example):
  ~~~
  $ python config.py stylegan --batch_size=16
  $ python data_config.py FFHQ path/to/datasets/ffhq --enable_mirror_augmentation
  $ python train.py
  ~~~

  [SAMPLES COMING SOON]

### ResNet GAN Example:

A ResNet GAN Generator can be created by running the following 3 lines (for example):
  ~~~
  $ python config.py resnetgan --lr_base=.00015
  $ python data_config.py LSUN-Bedrooms path/to/datasets/lsun_bedrooms
  $ python train.py
  ~~~

  [SAMPLES COMING SOON]



## Jupyter Notebook (or Custom Script) Usage

Running train.py is just the very basic usage. This package can be imported and utilized in a modular manner as well (like an API). For example, often it's helpful to experiment inside a Jupyter Notebook, like in the example below.

  First configure your GAN to your choosing on the command-line (like explained above under "Basic Usage on Command-line"):
  ~~~
  $ python config.py stylegan
  $ python data_config.py FFHQ path/to/datasets/ffhq
  ~~~

  Then write a custom script or Jupyter Notebook cells:
  ```python
  from gan_zoo import get_current_configuration
  from gan_zoo.utils.data_utils import prepare_dataset, prepare_dataloader
  from gan_zoo.stylegan.learner import StyleGANLearner

  # get most recent configurations:
  config = get_current_configuration( 'config' )
  data_config = get_current_configuration( 'data_config' )

  # get DataLoader(s)
  train_ds, valid_ds = prepare_dataset( data_config )
  train_dl, valid_dl, z_valid_dl = prepare_dataloader( config, data_config, train_ds, valid_ds )

  # instantiate StyleGANLearner and train:
  learner = StyleGANLearner( config )
  learner.train( train_dl, valid_dl, z_valid_dl )   # train for config.num_main_iters iterations
  learner.config.num_main_iters = 300000            # this is one example of changing your instantiated learner's configurations
  learner.train( train_dl, valid_dl, z_valid_dl )   # train for another 300000 iterations
  ```

__Some Advantages of Jupyter Notebook (there are many more than this)__:
+ You have the flexibility to think about what to do with your trained model after its trained rather than all at once, such as:
  + whether you want to save/load your trained model
  + what learner.config parameters you want to change before training again
+ You can always stop the kernel during training and then resume again and it will work



## TODO (will be implemented soon):
- [ ] Multi-GPU support
- [ ] TensorBoard capabilities
- [ ] FID, IS, and MS-SSIM metrics calculation
- [ ] Incorporate Spectral Normalization
- [ ] Incorporate Self-attention