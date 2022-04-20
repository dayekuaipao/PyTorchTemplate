# PyTorchTemplate
My PyTorch Template, can be easily embedded.

### lib

The code for the template.

##### lib/build

Something helpful for building the repo, for example, 'registry.py', it implements a registry for datasets, models and bckbones.

##### lib/datasets

The path of PyTorch datasets. There is an example 'cifar.py' for it.  Just like a traditional PyTorch dataset,but we should register it in dataset_registry.

##### lib/models

The path of models. There are some backbones such as inception, MobileNet, ResNet, VGG and loss function like focal loss in the directory. You can implement your own model using these backbones and loss.

##### lib/utils

The path for some utils. You can implement your learning rate scheduler (in 'lr_scheduler.py'), transforms (in 'transform.py'). There are also some helpful tools such as evaluator for evaluating your model's performance, logger for saving logs in tensorboard, saver for saving the model.

### test.py

Python script for testing. The parameter information is in it.

### train.py

Python script for training. The parameter information is in it.



## Tutorials

1. Create two directories 'data' and 'experiment'.
2. Put cifar10 dataset in 'data/cifar10/'. 
3. Create a directory 'experiment/expriment1'.
4. Create a shell file named 'train.sh' in the directory:

```shell
python ../../train.py --dataset_path ../../data/cifar-10-batches-py/ --num_train 40000
```

5. Run the shell command by:

```shell
sh train.sh
```

We can see a directory 'runs' (tensorboard files of training), and three files: 'best_checkpoint.pth', 'current_checkpoint' (to help resuming training process.), 'parameters.txt' (the parameter of training).

6. Create a shell file named 'test.sh' in the directory:

```python
python ../../test.py --dataset_path ../../data/cifar-10-batches-py/ --pretrained_model_path ./best_checkpoint.pth
```

7. Runn the shell command by:

```shell
sh test.sh
```

We can see a file 'confuse_matrix.csv' which is the confuse matrix of test data.



