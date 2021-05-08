# Fast and Effective Nasty Teaher
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

["Just Mildly Nasty: Fast and Effective Methods to Supress Student Model Learning"]

Sameer Bibikar, Matthew Kelleher



<!-- ## Overview TODO--> 

<!-- * TODO -->


## Prerequisite
We use Pytorch 1.4.0, and CUDA 10.1. You can install them with  
~~~
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
~~~   
It should also be applicable to other Pytorch and CUDA versions.  


Then install other packages by
~~~
pip install -r requirements.txt
~~~

## Usage 


### Teacher networks 

##### Step 1: Train a normal teacher network   

~~~
python train_scratch.py --save_path [XXX]
~~~
Here, [XXX] specifies the directory of `params.json`, which contains all hyperparameters to train a network.
We already include all hyperparameters in `experiments` to reproduce the results in our paper.    

For example, normally train a ResNet18 on CIFAR-10  
~~~
python train_scratch.py --save_path experiments/CIFAR10/baseline/resnet18
~~~
After finishing training, you will get `training.log`, `best_model.tar` in that directory.  
   
The normal teacher network will serve as the **adversarial network** for the training of the nasty teacher. 



##### Step 2: Train a nasty teacher network
For the original, fully nasty training use:
~~~
python train_nasty.py --save_path [XXX]
~~~

For half nasty training:
~~~
python train_half_nasty.py --save_path [XXX]
~~~
Whole network is retrainined, nasty KL divergence term is only applied to fully connected layers.

For the "mild" / "light" training:
~~~
python train_nasty_light.py --save_path [XXX]
~~~
Note that this training initalizes the network with the weights of the non-nasty adversarial network, freezes the convolutional layers, and then performs the nasty training on only the fully connected layers. 

For the "mild deep" / "light deep" training:
~~~
python train_nasty_light_deep.py --save_path [XXX] --layer_size [YYY]
~~~
* This nasty training option consists of initalizeing the convolution layer weights to the weights of the adversarial network and freezing them and then adding an additional fully connected layer to the network as described in the paper.
* The optional parameter --layer_size allows for the user to adjust the size of the added layer. If the parameter is omitted the layer size will default to the number of classes in the dataset. 

Again, [XXX] specifies the directory of `params.json`, 
which contains the information of adversarial networks and hyperparameters for training.  
You need to specify the architecture of adversarial network and its checkpoint in this file. 

 
For example, train a nasty ResNet18
~~~
python train_nasty.py --save_path experiments/CIFAR10/kd_nasty_resnet18/nasty_resnet18
~~~


### Knowledge Distillation for Student networks 

You can train a student distilling from normal or nasty teachers by 
~~~
python train_kd.py --save_path [XXX]
~~~
Again, [XXX] specifies the directory of `params.json`, 
which contains the information of student networks and teacher networks
 

For example,   
* train a plain CNN distilling from a nasty ResNet18 
~~~
python train_kd.py --save_path experiments/CIFAR10/kd_nasty_resnet18/cnn
~~~

* Train a plain CNN distilling from a normal ResNet18 
~~~
python train_kd.py --save_path experiments/CIFAR10/kd_normal_resnet18/cnn
~~~

NOTE: When distilling from a "mildly deep" / "light deep" model the `params.json` shoudl include an addition parameter, `teacher_deeper`, specifying the size of the added layer in the teacher model.

## Citation


## Acknowledgement
* [Nasty-Teacher](https://github.com/VITA-Group/Nasty-Teacher)
* [Teacher-free KD](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation)
* [DAFL](https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL) 
* [DeepInversion](https://github.com/NVlabs/DeepInversion)

