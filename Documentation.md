## Machine Learning Task: Fashion-MNIST

The task in hand is a typical image classification task, using the Fashion-MNIST dataset. Towards this end 3 different computationally efficient CNN architectures were implemented, trained and evaluated using various combinations of losses, optimisers, learning rates, batch sizes, normalisations and data augmentation

#### Classifier Architecture

Three architectures were implemented:

1. A simple CNN + FC architecture including 2 Conv5x5 / BN / ReLu / MaxPool layers followed by feature map flattening, FC / Relu / dropout and finally an FC. The architecture includes ~9M parameters, with almost all of them in the first FC layer (3136x3136)

2. The [Shufflenet V2](https://arxiv.org/abs/1807.11164) architecture, which is a SoA classifier using 3x3 depthwise separable convolutions along with channel shuffling and residual connections. It represents "handcrafted" efficient dw-based architectures (i.e Mobilenets etc.). The narrow x0.5 version is used, resulting in a very small model of just 382K parameters. In the last stage the feature maps are flattened and fed to an FC layer. 

3. The [MixNet S](https://arxiv.org/abs/1907.09595) architecture, which is a SoA classifier using multi-size depthwise separable convolutions and residual connections. It represents "AutoML search space" efficient dw-based architectures (i.e MNasNet, EfficientNet etc.). The slim S version is used, resulting in a small model of 2.6M parameters. In the last stage the feature maps are passed through an AveragePooling layer fed to a Dropout/FC layer.

#### Optimisers

Two optimisers are investigated, Adam and SGD with Nesterov momentum, with both optimisers having been succesfully employed in similar tasks in literature. Both optimizers achieve similar results, as expected, with sgd working slightly better for the simple CNN and MixNet, and Adam for Shufflenet. Three learning rates are investigated (0.1, 0.01, 0.001) as well as a learning rate decaying strategy (lr x0.1 every 10 epochs). 

#### Losses

Two losses are investigated:

1. Categorical Cross Entropy loss, which is the typical loss for such classification tasks

2. [Focal Loss](https://arxiv.org/abs/1708.02002) which is a custom loss based on CE which is used in the retinaNet object detector to improve results on highly unbalanced data, by reducing the impact of easy positives on the overall loss. Since the FashioMNIST dataset is well balanced, the simple CE outperformed Focal Loss, however it was an interesting route to explore 

#### Preprocessing

In terms of data preprocessing, normalising the data to [0-1] did produce a small improvement in results, however the batch norm layers helped with achieving very high accuracy results even without normalisation 

#### Data Augmentation

Fisrt Random horizontal flipping was employed. Next small randomisation of the image brightness was added, without however improving accuracy. Due to the relatively constrained structure of the dataset (i.e. all images are centered without any rotation), no further augmentation was explored, such as random cropping, rotation etc. which are commonly used in similar tasks.

#### Batch size

Three batch sizes were investigated (32, 64, 128), with 64 providing the best results

### Final results

Multiple combinations of the above architectures and parameters were trained from scratch for 30 epochs and evaluated. The simple CNN achieved the best result of 93.62%, closely followed by mixNet with 93.50%, with the extremelly small shufflenet further back at 91.91%.

Meanwhile, initializing shufflenet and mixNet from pretrained imagenet values further pushed accuracy to 92.72% and XX respectively

The detailed results from all the training sessions can be found [here](https://drive.google.com/file/d/1csOWy-xwY6Xk2VjNKZJIM93UozmKfq-_/view?usp=sharing).
