#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>
#include "math.h"
#include <iostream>
#include "stdio.h"
#include <vector>
#include <cassert>
#include <utility>


using namespace Eigen;


inline double sigmoid(const double z){
        // The sigmoid function for scalars
        return 1.0/(1.0+exp(-z));
}


inline VectorXd sigmoid(const VectorXd z){
        // The sigmoid function for VectorXd class

        VectorXd result(z.size());

        for (size_t i = 0; i < z.size(); i++) {
                result[i] = (1.0/(1.0+exp(-z[i])));
        }

        return result;
}

inline double sigmoid_prime(const double z){
        // Derivative of the sigmoid function
        return sigmoid(z)*(1.0-sigmoid(z));
}

inline VectorXd sigmoid_prime(const VectorXd z){
        // Derivative of sigmoid function for VectorXd class

        VectorXd result(z.size());

        for (size_t i = 0; i < z.size(); i++) {
                result[i] = sigmoid(z[i]) * (1.0 - sigmoid(z[i]));
        }
        return result;
}


inline VectorXd digit_vector(const int digit){

        VectorXd vec(10);

        for (size_t i = 0; i < 10; i++) {
                if(i == digit) vec[i] = 1.0;
                else {
                        vec[i] = 0.0;
                }
        }
        return vec;
}

inline int getPythonIndex(int idxIn, const int array_size){


  // Astoundingly bad implementation of the python negaitve indexing stuff
  // TODO: generalize!

// eg: if array_size == 3, idxIn = -3. Output: 0
// eg: if array_size == 3, idxIn = -2. Output: 1
// eg: if array_size == 3, idxIn =  2. Output: 2
// eg: if array_size == 3, idxIn = -4. Output: 1

  if(idxIn >= 0) return idxIn;
  else{
    if( (idxIn+array_size) >= 0) return idxIn+array_size;
    else{
      idxIn += array_size;
      if( (idxIn+array_size) >= 0) return idxIn+array_size;
    }
  }

}


// void writeWeightsBiasesToCSV(const std::string filename, NeuralNet& nnet){
//
//         std::ofstream myfile;
//         myfile.open (filename);
//
//         std::cout << "Writing weights and biases to csv...\n";
//
//
//         // write dimensions of neural network into first line in csv file:
//         myfile << nnet.m_nInput << "," << nnet.m_nHidden << "," << nnet.m_nOutput << std::endl;
//
//
//         //  write biases to file
//         for(int layer = 0; layer < 2; layer++) {
//                 for (int i = 0; i < nnet.m_biases[layer].size(); i++) {
//                         myfile << nnet.m_biases[layer](i) << ",";
//                         // std::cout << m_biases[layer](i) << ",";
//                 }
//                 myfile << std::endl;
//         }
//
//
//         //  write weights to file
//         for(int layer = 0; layer < 2; layer++) {
//                 for (int i = 0; i < m_weights[layer].rows(); i++) {
//                         for (int j = 0; j < m_weights[layer].cols(); j++) {
//                                 myfile << nnet.m_weights[layer](i,j) << ",";
//                                 // std::cout << m_weights[layer](i,j) << ",";
//                         }
//                 }
//                 myfile << std::endl;
//         }
//
//         myfile.close();
//
//
//         std::cout << "    ...writing to csv completed.\n";
//
//         return;
// }
//
// void readWeightsBiasesFromCSV(const std::string filename, NeuralNet& nnet){
//
//
//   std::vector<int> dims;
//   bool firstLine = true;
//
//
//   std::vector <std::vector <double> > data;
//   std::ifstream infile( filename );
//
//
//   while (infile){ // iterate through lines
//
//     std::string line;
//     if (!getline( infile, line )) break;
//
//     std::istringstream linestream( line );
//     std::vector <double> record;
//     std::string item;
//
//     while (linestream){ // iterate through single line, items separated by comma
//       if (firstLine){
//         if (!getline( linestream, item, ',' )) break;
//         dims.push_back( stoi(item) );
//         firstLine = false;
//       }
//       else{
//         if (!getline( linestream, item, ',' )) break;
//         record.push_back( stoi(item) );
//       }
//     }
//     data.push_back( record );
//   }
//
//   // now that we have all data in the data vector<vector<double>>, we need to reconstruct the
//   // weights and biases from it:
//
//   // data[0] : m_biases[0];
//   // data[1] : m_biases[1];
//   // data[2] : m_weights[0]
//   // data[3] : m_weights[1]
//   int lineNo = 0;
//
//   for(int i = 0; i < dims.size()-1; i++){
//     VectorXd bias(dims[i+1]);
//
//     for (int j = 0; j < dims[i+1]; j++) bias(j) = data[lineNo][j];
//
//     nnet.m_biases.push_back(bias);
//     lineNo++;
//   }
//
//
//   for(int i = 0; i < dims.size()-1; i++){
//     MatrixXd weight(dims[i],dims[i+1]);
//     for (int ii = 0; ii < dims[i]; ii++){
//       for (int jj = 0; jj < dims[i+1]; jj++){
//         int idx = ii + dims[i]*jj;
//         weight(ii,jj) = data[i+lineNo][idx];
//       }
//     }
//     nnet.m_weights.push_back(weight);
//   }
//
//
//
// }



#endif
