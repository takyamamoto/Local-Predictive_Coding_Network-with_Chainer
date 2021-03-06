# Deep Local Predictive Coding Network with Chainer
Deep Local Predictive Coding Network (**Local PCN**) implemented with Chainer

## Description
Kuan Han, *et al.*, "Deep Predictive Coding Network with Local Recurrent Processing for Object Recognition", [(arxiv)](https://arxiv.org/abs/1805.07526)  
**I am not the author of this paper.**

### Dataset
Use CIFAR10 or CIFAR100 datasets.

## Requirement
- Python >=3.6
- Chainer 4.0
- matplotlib

## Usage
Chainer model is defined with `network.py`

### Train
When you use GPU, please run following command.
```
python train.py -g 0 -e 100 -d cifar10 -t 5
```
## Results

### Loss
<img src="https://github.com/takyamamoto/Local-Predictive_Coding_Network-with_Chainer/blob/master/results/loss.png" width=70%>

### Accuracy
<img src="https://github.com/takyamamoto/Local-Predictive_Coding_Network-with_Chainer/blob/master/results/accuracy.png" width=70%>

## Model
The following image is a computational graph of one Recurrent Block when `LoopTimes` equals **2**.  
<img src="https://github.com/takyamamoto/Local-Predictive_Coding_Network-with_Chainer/blob/master/results/block.png" width=20%>

### Plot computational graph of the model
**First**, You have to install `Graphviz`. If you use Anaconda, you can install with next command.
```
conda install graphviz
```

**Second**, Run `train.py` with a following command.
```
python train.py -g 0 -e 1 -t 2
```
Then, `cg.dot` is generated under `./results/`.  

**Third**, Convert dot file to png file with next command.
```
dot -Tpng results/cg.dot -o cg.png
```
*You should not plot graph when LoopTimes equals 5 because graph file is too large*
