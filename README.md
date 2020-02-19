# GAN Zoo

<img align="center" src="https://github.com/sidward14/gan-zoo/raw/master/examples/for_readme/stylegan/stylemixed-grid_sample.png" height="696" width="900"/>

### _Higher Resolutions coming once model finishes training in Google Colab with 16 GB GPU Memory (these are 128x128 from a 6 GB GPU)_

__Currently supports:__
+ StyleGAN (https://arxiv.org/pdf/1812.04948.pdf)
+ ProGAN (https://arxiv.org/pdf/1710.10196.pdf)
+ ResNet GANs

Each GAN model's default settings emulates its most recent official implementation, but at the same time this package features a simple interface ([config.py](./gan_zoo/config.py)) where the user can quickly tune an extensive list of hyperparameter settings to his/her choosing.

Comes with additional features such as supervised learning capabilities, flexible learning rate scheduling (and re-scheduling) capabilities, an easy-to-use interface for saving/loading the model/learner state (e.g. for transfer learning or for evaluation at a later time), etc.

This package aims for an intuitive API without sacrificing any complexity anywhere.

--------------------------------------------------------------------------------

In your virtual environment (e.g. a conda virtual environment), run:
  ~~~
  $ pip install gan-zoo
  ~~~
This will install all necessary dependencies for you and will enable the option to use the package like an API (see "Jupyter Notebook (or Custom Script) Usage" below).

If you do not wish to use the package like an API (i.e. you just want to install dependencies and then just use the repo by means of running [train.py](./gan_zoo/train.py)), run '$ pip install -r requirements.txt' instead.

## Basic Usage on Command-line

__Clone this repo__, then simply run the following to configure your model & dataset and train your chosen model:
  ~~~
  $ python config.py [model] [--optional_kwargs]
  $ python data_config.py [dataset] [dataset_dir] [--optional_kwargs]
  $ python train.py
  ~~~
If you would like to see a list of what each argument does, run '$ python config.py [model] -h' or '$ python data_config.py [dataset] [dataset_dir] -h' on the command-line.

__NOTE__: Make sure that all images you would like to use in your model are located directly inside the _dataset_dir_ parent directory before running [data_config.py](./gan_zoo/data_config.py). Any images within subdirectories of _dataset_dir_ (except for the subdirectories named "train" or "valid" that get created when you run [data_config.py](./gan_zoo/data_config.py)) will not be used when training your model.

### StyleGAN Example:

A StyleGAN Generator that yields 128x128 images _(higher resolutions coming once model is done training in Google Colab with 16 GB GPU Memory!)_ can be created by running the following 3 lines. Below is a snapshot of images as the StyleGAN progressively grows. Ofcourse, this is not the only configuration that works:
  ~~~
  $ python config.py stylegan --loss=nonsaturating --gradient_penalty=R1 --res_samples=128 --num_main_iters=1071000 --nimg_transition=630000 --batch_size=8 --enable_cudnn_autotuner --num_workers=12
  $ python data_config.py FFHQ path/to/datasets/ffhq --enable_mirror_augmentation
  $ python train.py
  ~~~

  <p align="center">
  <img align="center" src="https://media.giphy.com/media/KCeh2byAXR6s0lrymS/giphy.gif?raw=true" width="500" height="500"/>
  </p>

By default, image grids like the one above are saved periodically during training into the "./gan_zoo/samples" directory every 1,000 iterations (see [config.py](./gan_zoo/config.py)). _Please note that the gif above contains a reduced number of frames compared to its original form (in order to reduce its file size)._

### ProGAN Example:

A ProGAN Generator that yields 128x128 images _(higher resolutions coming once model is done training in Google Colab with 16 GB GPU Memory!)_ like the ones below can be created by running the following 3 lines. Ofcourse, this is not the only configuration that works:
  ~~~
  $ python config.py progan --res_samples=128 --num_main_iters=1050000 --batch_size=8
  $ python data_config.py CelebA-HQ path/to/datasets/celeba_hq --enable_mirror_augmentation
  $ python train.py
  ~~~

  <p align="center">
  <img align="center" src="https://giphy.com/embed/KCeh2byAXR6s0lrymS?raw=true" width="500" height="500"/>
  </p>

By default, image grids of generator output are saved periodically during training into the "./gan_zoo/samples" directory every 1,000 iterations (see [config.py](./gan_zoo/config.py)).

### ResNet GAN Example:

A ResNet GAN Generator can be created by running the following 3 lines (for example):
  ~~~
  $ python config.py resnetgan --lr_base=.00015
  $ python data_config.py LSUN-Bedrooms path/to/datasets/lsun_bedrooms
  $ python train.py
  ~~~

  [SAMPLES FOR RESNET GAN COMING SOON]



## Jupyter Notebook (or Custom Script) Usage

Running [train.py](./gan_zoo/train.py) is just the very basic usage. This package can be imported and utilized in a modular manner as well (like an API). For example, often it's helpful to experiment inside a Jupyter Notebook, like in the example below.

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
  + whether you want to save/load your trained model/learner
  + what learner.config parameters you want to change before training again
+ You can always stop the kernel during training and then resume again and it will work

--------------------------------------------------------------------------------

__NOTE__ that by default, the _--num_workers_ argument in [config.py](./gan_zoo/config.py) is set to data-loading from just 1 subprocess; setting this to a larger number (that still falls within the constraints of your CPU(s)) will speed up training significantly. :slightly_smiling_face:

## TODO (will be implemented soon):
- [ ] Multi-GPU support
- [ ] TensorBoard capabilities
- [ ] FID, IS, and MS-SSIM metrics calculation
- [ ] Incorporate Spectral Normalization
- [ ] Incorporate Self-attention