#include "NeuralNet.h"


NeuralNet::NeuralNet(int nInput, int nHidden, int nOutput) {        // constructor

        m_nInput = nInput;
        m_nHidden = nHidden;
        m_nOutput = nOutput;

        dims = {m_nInput, m_nHidden, m_nOutput};
        std::cout << "Neural Network Dimensions: ";
        std::cout << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;


        VectorXd biases_1 = VectorXd::Random(m_nHidden);         // uniform between -1 and 1
        VectorXd biases_2 = VectorXd::Random(m_nOutput);         // uniform between -1 and 1
        m_biases.push_back(biases_1);
        m_biases.push_back(biases_2);


        std::cout << "Initializing random weights" << std::endl;


        MatrixXd weights_1 = MatrixXd::Random(m_nHidden,m_nInput);         // uniform between -1 and 1
        MatrixXd weights_2 = MatrixXd::Random(m_nOutput,m_nHidden);        // uniform between -1 and 1

        m_weights.push_back(weights_1);
        m_weights.push_back(weights_2);


	std::cout << "m_weights dimensions: " << m_weights[0].cols() << " " << m_weights[0].rows() << " " << m_weights[1].cols() << " " << m_weights[1].rows() << std::endl;

	std::cout << "m_biases dimensions: " << m_biases[0].size() << " " << m_biases[1].size() << std::endl;

}


void NeuralNet::SGD( Data trainingData,const int nEpochs,const int mini_batch_size,const double learningRate,const Data validationData){

        // """Train the neural network using mini-batch stochastic
        //     gradient descent.  The ``training_data`` is a list of tuples
        //     ``(x, y)`` representing the training inputs and the desired
        //     outputs.  The other non-optional parameters are
        //     self-explanatory.  If ``test_data`` is provided then the
        //     network will be evaluated against the test data after each
        //     epoch, and partial progress printed out.  This is useful for
        //     tracking progress, but slows things down substantially."""


        int nValSets = validationData.nEntries;
        int nTrainSets = trainingData.nEntries;

        std::vector<VectorXd> mini_batches_img;
        std::vector<int> mini_batches_digits;

        for(size_t j=0; j < nEpochs; j++){

              std::cout << "Epoch " << j << " started..." <<  std::endl;
              int dataSetSize = 5000;
              // int nMiniBatches = dataSetSize / mini_batch_size;

              // need random indices from 0 to dataSetSize-1
              std::vector<int> randomIndices;
              int max = dataSetSize-1-mini_batch_size; // must have mini_batch_size distance from last index
              int min = 0;
              for (size_t i = 0; i < dataSetSize; i++) {
                randomIndices.push_back(rand()%(max-min + 1) + min); // works
              }

              mini_batches_img.clear();
              mini_batches_digits.clear();



              for (size_t i = 0; i < dataSetSize; i++){
                    mini_batches_img.push_back(trainingData.getImgEntry(randomIndices[i]));
                    mini_batches_digits.push_back(trainingData.getDigitEntry(randomIndices[i]));
              }


              for(size_t i = 0; i < dataSetSize; i+= mini_batch_size){
                  if(i % 500 == 0) std::cout << "        " << i << " of " <<  dataSetSize << std::endl;
                  update_mini_batch(mini_batches_img,mini_batches_digits, i,learningRate,mini_batch_size); // update weigts and biases
              }

              std::cout << "Epoch " << j << ": " << evaluate(validationData) << " / " << nValSets << std::endl;

        }

}

VectorXd NeuralNet::feedforward(const VectorXd input){

        // feed to hidden layer
        VectorXd temp1 = m_weights[0] * input;
        VectorXd middle = sigmoid(temp1+ m_biases[0]);

        // feed to output layer
       VectorXd temp2 = m_weights[1] * middle;
       VectorXd out = sigmoid(temp2 + m_biases[1]);

        return out;
}




VectorXd NeuralNet::cost_derivative_cross_entropy(const VectorXd output_activations,const VectorXd digitVec,const VectorXd z){ // cross-entropy cost function

        return output_activations - digitVec;
}

VectorXd NeuralNet::cost_derivative_quad( VectorXd output_activations, VectorXd digitVec, VectorXd z){ //  quadratic cost function

  VectorXd tmp = output_activations - digitVec;
  VectorXd delta = tmp.array() * sigmoid_prime(z).array(); // conversion to array in order to perform element-by-element vector multiplication

  return delta;
}


void NeuralNet::update_mini_batch(const std::vector<VectorXd> images,const std::vector<int> digits,const int data_idx,const double learningRate,const int mini_batch_size){
  // """Update the network's weights and biases by applying
  // gradient descent using backpropagation to a single mini batch.


  std::vector<std::vector<MatrixXd>> delta_nabla_ws;
  std::vector<std::vector<VectorXd>> delta_nabla_bs;

  std::vector<MatrixXd> delta_nabla_w;
  std::vector<VectorXd> delta_nabla_b;


  for (size_t offs = 0; offs < mini_batch_size; offs++){ // go thorugh mini batch

    int idx = data_idx + offs;

    std::pair<  std::vector<MatrixXd> , std::vector<VectorXd>  > delta_nablas = backprop(images[idx],digits[idx]);

    delta_nabla_w = delta_nablas.first;
    delta_nabla_b = delta_nablas.second;

    delta_nabla_ws.push_back(delta_nabla_w);
    delta_nabla_bs.push_back(delta_nabla_b);
  }


  std::vector<MatrixXd> nabla_w;
  std::vector<VectorXd> nabla_b;


    // preallocate nabla_b and nabla_w with zeros
  nabla_w.push_back(MatrixXd::Zero(m_nHidden,m_nInput));
  nabla_b.push_back(VectorXd::Zero(m_nHidden));
  nabla_w.push_back(MatrixXd::Zero(m_nOutput,m_nHidden));
  nabla_b.push_back(VectorXd::Zero(m_nOutput));


  // add single delta_nabla_ws to nabla_w
  for(int i = 0; i < 2; i++){
    for(size_t idx = 0; idx < mini_batch_size; idx++){
       nabla_b[i] += delta_nabla_bs[idx][i];
    }
  }


   // add single delta_nabla_bs to nabla_b
   for(int i = 0; i < 2; i++){
     for(size_t idx = 0; idx < mini_batch_size; idx++){
        nabla_w[i] += delta_nabla_ws[idx][i];
     }
   }



// update weights and biases


    for (int i = 0; i < 2; i++){
      VectorXd tmp1 = (learningRate/(double)mini_batch_size) * nabla_b[i];
      m_biases[i]  -= tmp1;

      MatrixXd tmp2  =  (learningRate/(double)mini_batch_size) * nabla_w[i];
      m_weights[i]  -=  tmp2;
    }

}


int NeuralNet::evaluate( Data validationData){

  // """Return the number of test inputs for which the neural
  // network outputs the correct result. Note that the neural
  // network's output is assumed to be the index of whichever
  // neuron in the final layer has the highest activation."""

    int sum = 0;


    for(size_t i = 0; i < validationData.nEntries; i++){ // go through all validationData entries
        VectorXd out = feedforward(validationData.getImgEntry(i));

        // find index of max element
        int maxIdx = 0;
        for(int i = 0; i < out.size(); i++){
          if(out[i] > out[maxIdx]) maxIdx = i;
        }

        int predictedDigit = maxIdx;
        int actualDigit = validationData.getDigitEntry(i);

        if (actualDigit == predictedDigit) sum += 1;
    }


    return sum;

}



std::pair<  std::vector<MatrixXd> , std::vector<VectorXd>  > NeuralNet::backprop(const VectorXd img,const int digit){

  // def backprop(self, x, y):
  // """Return a tuple ``(nabla_b, nabla_w)`` representing the
  // gradient for the cost function C_x.  ``nabla_b`` and
  // ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
  // to ``self.biases`` and ``self.weights``."""

  std::vector<MatrixXd> nabla_w;
  std::vector<VectorXd> nabla_b;


  VectorXd                  activation = img;
  std::vector<VectorXd>     activations;

  activations.push_back(activation);

  VectorXd                    z;
  std::vector<VectorXd>       zs;

  for(size_t i = 0; i <= 1; i ++){
    z = m_weights[i] * activation;
    zs.push_back(z);
    activation = sigmoid(z);
    activations.push_back(activation);

  }


   // Backward pass from output layer to hidden layer

  VectorXd delta1 = cost_derivative_cross_entropy(activations[2], digit_vector(digit), zs[1]);

  MatrixXd mult_result1 = activations[1] * delta1.transpose();


  // Backward pass from hidden layer to input layer

   VectorXd sp = sigmoid_prime(zs[0]);
   VectorXd tmp = m_weights[1].transpose() * delta1;
   VectorXd delta2 = tmp.array() * sp.array();


   MatrixXd mult_result2 = activations[0] * delta2.transpose();


   nabla_b.push_back(delta2);
   nabla_b.push_back(delta1);

   nabla_w.push_back(mult_result2.transpose());
   nabla_w.push_back(mult_result1.transpose());

  std::pair<std::vector<MatrixXd>,std::vector<VectorXd>> result = make_pair(nabla_w,nabla_b); // declare output std::pair

  return result;


}


void NeuralNet::writeWeightsBiasesToCSV(const std::string filename){

  std::ofstream myfile;
  myfile.open (filename);

  //  write weights to file

  std::cout << "writing weights to file\n";

  for(int layer = 0; layer < 2; layer ++){
    for (int i = 0; i < m_weights[layer].rows(); i++){
      for (int j = 0; j < m_weights[layer].cols(); j++){
        myfile << m_weights[layer](i,j) << ",";
        std::cout << m_weights[layer](i,j) << ",";

      }
    }
    myfile << std::endl;
  }



    std::cout << "writing biases to file\n";

  //  write biases to file
  for(int layer = 0; layer < 2; layer ++){
    for (int i = 0; i < m_biases[layer].size(); i++){
      myfile << m_biases[layer](i) << ",";
      std::cout << m_biases[layer](i) << ",";
      }
    myfile << std::endl;
    }
  myfile.close();


    std::cout << "writing finished\n";

  return;
}



NeuralNet::~NeuralNet(){   // empty destructor

}
