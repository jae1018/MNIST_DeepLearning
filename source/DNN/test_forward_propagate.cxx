#include <gtest/gtest.h>
#include "DNN.h"

/**
* Tests the forward propagation of the DNN with...
* (1) Uniform weights and biases (everything = 1) and uniform input (= 1), and
* (2) Non-uniform weights and biases (value determined based on receiving node)
* and uniform input (= 1).
*
* This test suite requires the DNN to be structured with 5
* layers and number of nodes = {5, 4, 3, 2, 2}. It also assumes a
* sigmoid activation function.
*
* @author: James "Andy" Edmond
* @date: April 27, 2020
*/

TEST(ForwardPropTest, UniformWeightsAndBiases_UniformInput) {
  DNN network_test = DNN("this_is_a_bad_path.csv");
  int num_layers = 5;
  int layer_sizes[5] = {5, 4, 3, 2, 2};
  // confirm sizes are right
  xt::xtensor<int,1> sizes = network_test.get_layer_sizes();
  for (int i = 0; i < network_test.get_num_layers(); i++) {
    assert( layer_sizes[i] == sizes(i) );
  }

  // set weights and biases to uniform
  for (int i = 1; i < num_layers; i++) {
    arr weights = network_test.get_weights(i);
    weights.fill(1.0);
    network_test.set_weights(i,weights);
    vec biases = network_test.get_biases(i);
    biases.fill(1.0);
    network_test.set_biases(i,biases);
  }

  // make input vec
  vec input = xt::zeros<double>({layer_sizes[0]});
  for (int a = 0; a < layer_sizes[0]; a++) {
    input(a) = 1.0;
  }

  // do computations
  network_test.forward_propagate(input);

  // Then get output and compare to expected result
  vec output = network_test.get_activations(num_layers-1);
  vec answer = xt::zeros<double>({layer_sizes[num_layers - 1]});
  answer.fill(0.95089);
  const double tol = 0.01;
  for (int i = 0; i < answer.size(); i++) {
    double diff = abs(answer(i) - output(i));
    ASSERT_TRUE(diff < tol);
  }
}


TEST(ForwardPropTest, NonUniformWeightsAndBiases_UniformInput) {
  DNN network_test = DNN("this_is_a_bad_path.csv");
  int num_layers = 5;
  int layer_sizes[5] = {5, 4, 3, 2, 2};
  // confirm sizes are right
  xt::xtensor<int,1> sizes = network_test.get_layer_sizes();
  for (int i = 0; i < network_test.get_num_layers(); i++) {
    assert( layer_sizes[i] == sizes(i) );
  }

  // set weights and biases to to int of recv_node / 10
  for (int i = 1; i < num_layers; i++) {
    arr weights = network_test.get_weights(i);
    for (int row_num = 0; row_num < weights.shape()[0]; row_num++) {
      for (int col_num = 0; col_num < weights.shape()[1]; col_num++) {
        weights(row_num,col_num) = double(col_num)/10;
      }
    }
    network_test.set_weights(i,weights);
    vec biases = network_test.get_biases(i);
    for (int node_num = 0; node_num < biases.size(); node_num++) {
      biases(node_num) = double(node_num)/10;
    }
    network_test.set_biases(i,biases);
  }

  // make input vec
  vec input = xt::zeros<double>({layer_sizes[0]});
  for (int a = 0; a < layer_sizes[0]; a++) {
    input(a) = 1.0;
  }

  // do computations
  network_test.forward_propagate(input);

  // compare output to guess
  vec output = network_test.get_activations(num_layers-1);
  vec answers = {0.5, 0.55154};
  const double tol = 0.001;
  for (int i = 0; i < answers.size(); i++) {
    double diff = abs(output(i) - answers(i));
    ASSERT_TRUE(diff < tol);
  }
}
