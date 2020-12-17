# PyTorchTemplate
My PyTorch Template, can be easily embedded.

### lib

The code for the template.

##### lib/build

Something to help to build the repo, for example, 'registry.py', it implements a registry for datasets, models and bckbones.

##### lib/datasets

The path of PyTorch datasets. There is a example 'cifar.py' for it.  Just like a traditional PyTorch dataset,but we should register it in dataset_registry.

##### lib/models

The path of models. There are some backbones such as  inception, MobileNet, ResNet, VGG in the directory. You can implement your own model using these backbones.

##### lib/utils

The path for some utils. You can implement your own loss (in 'loss.py'), learning rate scheduler (in 'lr_scheduler.py'), transforms (in 'transform.py'). There are also some helpful tools such as evaluator for evaluating your model's performance, logger for saving logs in tensorboard, saver for saving the model.

### test.py

Python script for testing. The parameter information is in it.

### train.py

Python script for training. The parameter information is in it.



## Examples

We first create two directories 'data' and 'experiment'. We put cifar10 dataset in 'data/cifar10/'. 

Then we create a directory 'experiment/expriment1'. We  create a shell file named 'train.sh' in the directory:

```shell
python ../../train.py --dataset_path ../../data/cifar-10-batches-py/ --num_train 40000
```

Then we run the shell command by:

```shell
sh train.sh
```

We can see a directory 'runs' (tensorboard files of training), and three files: 'best_checkpoint.pth', 'current_checkpoint' (to help resuming training process.), 'parameters.txt' (the parameter of training).

Next we  create a shell file named 'test.sh' in the directory:

```python
python ../../test.py --dataset_path ../../data/cifar-10-batches-py/ --pretrained_model_path ./best_checkpoint.pth
```

Then we run the shell command by:

```shell
sh test.sh
```

At last e can see a file 'confuse_matrix.csv' which is the confuse matrix of test data.



