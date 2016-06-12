#ifndef DATA_H
#define DATA_H

#include "string.h"
#include "util.h"
#include <Eigen/Dense> // eigen library for matrix vector stuff
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>

using namespace Eigen;

class Data {



// Structure of Data Class Instantiation:
// MNIST raw data: 70,000 Data sets
// Useful splitting:
// training data:       60,000 entries
// validation data:     10,000 entries
// structure of each entry:
// (x,y) tuple, while:
//    x: 28x28 (784-dimensional) vector of 8bit(0-255) grey values
//    y: 10-dimensional binary vector of actual digit depicted


public:
        void load_data_from_file(const std::string filename);
        Data();                                     // empty constructor
        size_t nEntries;                            // number of entries in data set
        int getDigitEntry(int index);
        VectorXd getImgEntry(int index);

private:

        std::vector<VectorXd> imgData;
        std::vector<int> correctDigit;

};

#endif
