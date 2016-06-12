### C++ Neural Network for digit recognizing. Application on the MNIST data set
### VERSION 2: Incorporates more sophisticated programming stuff

## Requirements

- [eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page) header files available
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



## TODO: Style

- add description text to each function
- clean up code. A lot!
- const correctness


## TODO: Speedup

- use openMP to parallelize learning
- pointer?

## TODO: More sophisticated

- implement writeWeightsBiasesToCSV method
- implement cross-entropy cost function
- implement convolutionary neural networks
