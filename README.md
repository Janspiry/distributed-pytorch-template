# Pytorch Template Using DistributedDataParallel

This repository is a seed project for distributed training using pytorch, was built to customize datasets, networks, parameters and hyper-parameters quickly during the training and test. Every part can be modifications easily to build your network. 

------
### Basic Functions

- checkpoint/resume training
- progress bar (using tqdm)
- progress logs (using logging)
- progress visualization (using tensorboard)
- finetune (partial network parameters training)
- learning rate scheduler
- random seed (reproducibility)

### Features

- distributed training using DistributedDataParallel
- base class for more expansibility
- `.json` config file for most parameter tuning.
- support multi networks/losses/metrics definition
- debug mode for fast test

------
### Usage

#### Start

Run the `run.py` with your setting.

```python
python run.py
```

More choices can be found on `run.py` and `config/base.json`.

*Note: cuDNN default settings are as follows for training, which may reduce your code reproducibility! Notice it to avoid the unexpected behaviors.*

```python
 torch.backends.cudnn.enabled = True
 # speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
 if seed >=0 and gl_seed>=0:  # slower, more reproducible
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
 else:  # faster, less reproducible, default setting
     torch.backends.cudnn.deterministic = False
     torch.backends.cudnn.benchmark = True
```

#### Customize Dataset

Dataset part decide the data need to be fed into network, you can define your dataset by following steps:

1. Put your dataset under `data` folder. See `dataset.py` in this folder as an example.
2. Edit the **\[dataset\]\[train|test\]** part in `config/base.json` to import and initialize your dataset. 

```json
"datasets": { // train or test
    "train": { 
		"which_dataset": {  // import designated dataset using args 
            "name": ["data.dataset", "Dataset"], // import Dataset() class from data.dataset.py (default is [data.dataset.py])
            "validation_split": 0.1, // percent or number
            "args":{ // args to init dataset
                "data_root": "/home/huangyecheng/dataset/cmfd/comofod"
                // "data_root": "/data/jlw/datasets/comofod"
            } 
        },
        "dataloader":{
            "args":{ // args to init dataloader
                "batch_size": 2, // batch size in every gpu
                "num_workers": 4,
                "shuffle": true,
                "pin_memory": true,
                "drop_last": true
            }
        }
    },
}
```

##### More details

- You can create your new file. Key `name` can be a list to show your file name and class/function name, or single string to explain class name in default folder (`data.dataset`), example is as follows:

```json
"name": ["data.dataset", "Dataset"], // import Dataset() class from data.dataset.py
"name": "Dataset", // import Dataset() class from default file
```

- You can control and record more parameters through config file. Take `data_root`  as the example, you just need to add it in args dict and edit corresponding class to parse this value:

```json
"which_dataset": {  // import designated dataset using args 
    "args":{ // args to init dataset
        "data_root": "/home/huangyecheng/dataset/cmfd/comofod"
    } 
},
```

```python
class Dataset(data.Dataset):
    def __init__(self, data_root, phase='train', image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root) # data_root value is from config file
```



#### Customize Network

Network part shows your learning network structure, you can define your network by following steps:

1. Put your network under `models` folder. See `network.py` in this folder as an example.
2. Edit the **\[model\][which_networks]** part in `config/base.json` to import and initialize your networks, and it is a list. 

```json
"which_networks": [ // import designated list of networks using args
    {
        "name": "Network", // import Network() class / function(not recommend) from default file (default is [models/network.py]) 
        "args": { // args to init network
            "init_type": "kaiming" // method can be [normal | xavier| xavier_uniform | kaiming | orthogonal], default is kaiming
        }
    }
],
```
##### More details

- You can create your new file. Key `name` can be a list to show your file name and class/function name, or single string to explain class name in default folder (`models.network` ), example is as follows:

```json
"name": ["models.network", "Network"], // import Network() class from models.network.py
"name": "Network", // import Network() class from default file
```

- You can control and record more parameters through config file. Take `init_type`  as the example, we just need to add it in args dict and edit corresponding class to parse this value:

```json
"which_networks": [ // import designated list of networks using args
    {
        "args": { // args to init network
            "init_type": "kaiming" 
        }
    }
],
```

```python
class BaseNetwork(nn.Module):
  def __init__(self, init_type='kaiming', gain=0.02):
    super(BaseNetwork, self).__init__() # init_type value is from config file
class Network(BaseNetwork):
    def __init__(self, in_channels=3, **kwargs):
        super(Network, self).__init__(**kwargs) # get init_type value and pass it to base network
```




#### Customize Model(Trainer)

Model part shows your training process including optimizers/losses/process control, etc.  You can define your model by following steps:

1. Put your Model under `models` folder. See `model.py` in its folder as an example.
2. Edit the **\[model\][which_model]** part in `config/base.json` to import and initialize your model.

You can create new file for network and record more parameters through config file, please refer to `Customize Network/More details` part.

After above steps, you need to rewrite several functions like  `base_model.py/model.py` for your network and dataset.

##### Losses and Metrics

Losses and Metrics are defined on config file. You also can control and record more parameters through config file, please refer to `Customize Network/More details` part.

```json
"which_metrics": [ 
    "mae" // import mae() function/class from default file (default is [models/metrics.py]) 
], 
"which_losses": [
    "mse_loss" // import mse_loss() function/class from default file (default is [models/losses.py]) 
] 
```

##### Training/validation step

See `save_everything()/load_everything()` functions as the example.

##### Checkpoint/Resume training

See `train_step()/val_step()` functions as the example.

##### Save Results

See `save_current_results()` functions as the example.



#### Debug mode

Sometime we hope debugging the process quickly to to ensure the whole project works, so debug mode appears. 

It will reduce the dataset size and speed up the training process. You just need to run the file with `-d` option                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           and edit the `debug` dict in config file.

```python
python run.py -d
```

```json
"debug": { // args in debug mode, which will replace args in train
    "val_epoch": 1,
    "save_checkpoint_epoch": 1,
    "log_iter": 30,
    "data_len": 50 // change the size of dataloder to data_len.
}
```



#### Customize More 

You can choose the random seed,  experiment path in config file. We will add more useful basic functions with related instructions. **Welcome to more contributions for more extensive customization and code enhancements.**

------
### Todo

Here are some basic functions or examples that this repository is ready to implement:

- [x] basic dataset/dataloader with validation split
- [x] basic networks with weight initialization
- [x] basic model (trainer)
- [x] checkpoint/resume training
- [x] progress bar (using tqdm)
- [x] progress logs (using logging)
- [x] progress visualization (using tensorboard)
- [x] multi-gpu support (using DistributedDataParallel and torch.multiprocessing)
- [x] finetune (partial network parameters training)
- [x] learning rate scheduler
- [x] random seed (reproducibility)


------
### Acknowledge

We are benefit a lot from following projects:

> 1. https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
> 2. https://github.com/researchmm/PEN-Net-for-Inpainting
> 3. https://github.com/tczhangzhi/pytorch-distributed
> 4. https://github.com/victoresque/pytorch-template