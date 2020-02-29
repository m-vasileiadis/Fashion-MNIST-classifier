## Machine Learning Task: Fashion-MNIST

Fashion-MNIST classifier implemented in Pytorch

#### Architectures:

1. Simple CNN: 2x Conv / BN / Relu / MP + FC / Relu / Dropout + FC 
2. [Shufflenet V2 x0.5](https://arxiv.org/abs/1807.11164) + flatten / FC
3. [MixNet S](https://arxiv.org/abs/1907.09595) + AP / Dropout / FC

#### Losses:
1. CrossEntropy Loss
2. [Focal Loss](https://arxiv.org/abs/1708.02002)

#### Optimisers
1. SGD + momentum + nesterov
2. Adam

#### Preprocessing
1. None
2. Normalisation to [0,1]

#### Augmentation
1. Random horizontal Flipping
2. Random image intensity 

### Results

Multiple combinations of the architectures, losses and optimisers were trained and evaluated, for different batch sizes (32, 64, 128), and learning rate values (0.1, 0.01, 0.001, constant/decaying x0.1 every 10 epochs). All experiments ran for 30 epochs

The best performing Configurations for the three architectures were:

| Arch | Loss | Optim | lr | Batch | Prep | Aug | Accuracy | Params | Input dim | MACS |Runtime (GTX 970) | Runtime (i5-4670) | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |--- |--- |
|Simple CNN | CE | SGD | 0.01d | 64 | [0-1] | flip | 93.62 | 9921 K | 28x28 | 20.7 M |0.54ms |2.71ms |
|Shufflenet v2 | CE | ADAM | 0.001 | 64 | None | flip | 91.91 / 92.72* | 382 K | 56x56 |3.2 M |7.99ms |6.00ms |
|MixNet S | CE | SGD | 0.01d | 64 | [0-1] | flip | 93.50 | 2612 K | 56x56 | 19.5 M| 17.14ms |17.96ms |

*initialized from pretrained imagenet

The detailed accuracy results from all the training sessions/configurations are available [here](https://drive.google.com/file/d/1csOWy-xwY6Xk2VjNKZJIM93UozmKfq-_/view?usp=sharing)

### Usage
to train a model:
```python
python3 main.py -a {'conv','shufflenet','mixnet'} -c {'ce','fl'} -op {'sgd','adam'} -lr learning_rate -b batch_size --epochs epochs --normalise --aug_int
```
to benchmark the models:
```python
python3 benchmark.py
```
