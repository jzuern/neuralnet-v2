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

        void SGD( Data trainingData,const int nEpochs,const int mini_batch_size,const double learningRate,const Data validationData);
        VectorXd feedforward(const VectorXd input);
        VectorXd cost_derivative_quad(const VectorXd output_activations,const VectorXd digitVec,const VectorXd z); // quadratic cost function
        VectorXd cost_derivative_cross_entropy(const VectorXd output_activations,const VectorXd y,const VectorXd z); // cross-entropy cost function
        void update_mini_batch(const std::vector<VectorXd> images,const std::vector<int> digits,const int idx,const double learningRate,const int mini_batch_size);
        int evaluate( Data);
        std::pair<std::vector<MatrixXd>,std::vector<VectorXd>> backprop(const VectorXd img,const int digit);
        void writeWeightsBiasesToCSV(const std::string filename);

private:
        size_t m_nInput;
        size_t m_nHidden;
        size_t m_nOutput;
        std::vector<size_t> dims;
        size_t m_nLayers = 3;
        std::vector<VectorXd> m_biases;  // every entry of m_biases stores bias vector for specific layer
        std::vector<MatrixXd> m_weights; // every entry of m_weights stores weight matrix for specific layer

};


#endif
