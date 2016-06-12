#ifndef NEURALNET_H
#define NEURALNET_H

#include <Eigen/Dense> // eigen library for matrix vector stuff
#include <random>
#include "util.h"
#include <iostream>
#include <algorithm>
#include "Data.h"
#include <numeric>

using namespace Eigen;

class NeuralNet {

public:
        NeuralNet (int,int,int); // constructor

        ~NeuralNet();           // Destructor

        // Stochastic Gradient Descent learning algorithm
        void SGD(Data trainingData, int nEpochs, int mini_batch_size, double learningRate, Data validationData);

        VectorXd feedforward(VectorXd);

        VectorXd cost_derivative(VectorXd output_activations, VectorXd y);

        void update_mini_batch(std::vector<VectorXd> images, std::vector<int> digits, int idx, double learningRate);

        int evaluate(Data);

        std::pair<std::vector<MatrixXd>,std::vector<VectorXd>> backprop(VectorXd img, int digit);


        void writeWeightsBiasesToCSV(std::string filename);


private:
        size_t m_nInput;
        size_t m_nHidden;
        size_t m_nOutput;

        std::vector<size_t> dims;

        size_t m_nLayers = 3;

        std::vector<VectorXd> m_biases; // every entry of m_biases stores bias vector for specific layer

        std::vector<MatrixXd> m_weights; // every entry of m_weights stores weight matrix for specific layer

};


#endif
