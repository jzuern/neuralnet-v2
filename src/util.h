#ifndef UTIL_H
#define UTIL_H


#include <Eigen/Dense> // eigen library for matrix vector stuff
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
        // The sigmoid function for vectors

        VectorXd result(z.size());

        for (size_t i = 0; i < z.size(); i++){
          result[i] = (1.0/(1.0+exp(-z[i])));
        }

        return result;
}

inline double sigmoid_prime(const double z){
        // Derivative of the sigmoid function
        return sigmoid(z)*(1.0-sigmoid(z));
}

inline VectorXd sigmoid_prime(const VectorXd z){


        VectorXd result(z.size());

        for (size_t i = 0; i < z.size(); i++){
          result[i] = sigmoid(z[i]) * (1.0 - sigmoid(z[i]));
        }


        return result;
}


inline VectorXd digit_vector(const int digit){

  VectorXd vec(10);

  for (size_t i = 0; i < 10; i++){
    if(i == digit) vec[i] = 1.0;
    else {
      vec[i] = 0.0;
    }
  }
  return vec;

}




#endif
