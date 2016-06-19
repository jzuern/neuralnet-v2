### Implementation of a simple neural network for digit recognition in C++


## v2 incorporates more sophisticated concepts like parallelization and regularization

The neural network has been designed in order to process the MNIST data given as a raw pixel lists from a csv file.

It has one hidden layer with an arbitrary number of hidden neurons. I implemented the L2 regularization approach which reduces the risks of overfitting to the training data set.

The reinforcement learning technique is implemented using the backpropagation algorithm with a stochastic gradient descent algorithm. The training data set is hereby divided into mini batches, which are selected by a random number generator.

This work was inspired by the [great online-book Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. I reimplemented the therein developed  python implementations in C++. 



## Build requirements

- [eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [openMP](http://openmp.org/wp/)

## Build

compile with provided makefile

```
$ make
```

## Usage

Run with
```
$ ./nnet
```

The [MNIST csv data sets](http://pjreddie.com/projects/mnist-in-csv/) must be in a local directory.



## TO DO:

- use openMP to parallelize learning
