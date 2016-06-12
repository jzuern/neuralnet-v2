// include standard libraries header
#include <fstream>
#include <iostream>
#include "stdio.h"

// include header for the 2 classes
#include "NeuralNet.h"
#include "Data.h"


int main () {

        std::cout << "NeuralNet started\n";

        // instantiate NeuralNet object
        NeuralNet nnet(784, 50, 10);  // nInput, nHidden, nOutput

        // instantiate Data object
        std::string testDataDir = "/home/jzuern/Dropbox/develop/C++/Nnet/data/mnist_train.csv";
        std::string validationDataDir = "/home/jzuern/Dropbox/develop/C++/Nnet/data/mnist_test.csv";

        Data trainingData, validationData;                      // instantiate Data object
        trainingData.load_data_from_file(testDataDir);          // load data file into object memory
        validationData.load_data_from_file(validationDataDir);  // load data file into object memory


        const double learningRate = 3.0; // learning rate
        const int nEpochs =         30; // number of epochs
        const int mini_batch_size = 10; // size of mini batch

        // train NeuralNet with Stochastic Gradient Descent Method
        nnet.SGD(trainingData, nEpochs, mini_batch_size, learningRate, validationData);


        // write weights to file
        std::string filename = "data/weights+biases.csv";
        nnet.writeWeightsBiasesToCSV(filename);

        char c;
        std::cin >> c;

        return 0;
}
