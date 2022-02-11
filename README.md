# A-Seed-Project-For-Deep-Learning-by-Pytorch

This repository is a seed project for deep learning, was built to customize networks, datasets, parameters and hyper-parameters quickly during the training and test process, which based on the same multi-GPU training process with log recording. Every part can be modifications easily to build your network.

### Todo

Here are some basic functions or examples that this repository is ready to implement:

- [x] unified training step
- [x] unified validation/test step
- [x] checkpoint/resume training
- [x] progress bar (using tqdm)
- [x] progress logs (using logging)
- [x] progress visualization (using tensorboard)
- [x] multi-gpu support (using DistributedDataParallel and torch.multiprocessing)
- [x] network weight initialization
- [x] finetune (partial network parameters training)
- [x] learning rate scheduler
- [x] random seed
- [x] custom loss

------

### Usage

#### Start

Run the `run.py` with your setting.

```python
python run.py -d
```

More choices can be found on `run.py` and `config/base.json`.

#### Customize Network

Network part shows your learning network structure, you can define your network by following steps:

1. Put your network under `models/network` folder. See `ae.py` in this folder as an example.
2. Edit the **\[model\][which_networks]** part in `config/base.json` to indicates network file's name, and it can be a list with respective weight initialization methods. Your networks will be imported by **define_networks** function in `models/networks.py`.

#### Customize Dataset

Dataset part decide the data need to be fed into network, you can define your dataset by following steps:

1. Put your dataset under `data` folder. See `image_dataset.py` in this folder as an example.
2. Edit the **\[dataset\]\[train|val|test\][name]** part in `config/base.json` to indicates dataset file's name, then your dataset will be imported by **create_dataset** function in `data/__init__.py`.

#### Customize Model(Trainer)

Model part shows your training process including optimizers/loss/process control, etc.  You can define your model by following steps:

1. Put your Model under `models` folder. See `ae_model.py` in its folder as an example.
2. Edit the **\[model\][which_model]** part in `config/base.json` to indicates model file's name, then your model will be imported by **create_model** function in `model/__init__.py`.

After above steps, you need to rewrite several functions like  `ae_model.py` for your network and dataset.

##### Parameters

To customize your loss, optimizers, schedulers, etc. See `__init__()` functions as an example.

##### Inputs

See `set_input()` functions as an example, which is read from your dataset.

##### Training Steps

See `optimize_parameters/val()/test()` functions as the example.

##### Checkpoint/Resume training

See `load()/save()` functions as the example.

##### Results

See `get_current_log()/get_current_visuals()` functions as the example.

##### 

#### Customize More 

You can choose the random seed, custom image augmentations/weight initialization in corresponding folder. We will add more useful basic functions with related instructions. **Welcome to more contributions for more extensive customization and code enhancements.**



------

### Acknowledge

We are benefit a lot from following projects:

> 1. https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
> 2. https://github.com/researchmm/PEN-Net-for-Inpainting
> 3. https://github.com/tczhangzhi/pytorch-distributed
> 4. https://github.com/AAnoosheh/ComboGAN
> 5. https://github.com/hejingwenhejingwen/AdaFM