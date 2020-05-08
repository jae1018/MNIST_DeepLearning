#include <gtest/gtest.h>
#include "DNN.h"

/**
* Tests the forward propagation of the DNN with...
* (1) Uniform weights and biases (everything = 1) and uniform input (= 1), and
* (2) Non-uniform weights and biases (value determined based on receiving node)
* and uniform input (= 1).
*
* This test suite requires the DNN to be structured with 5
* layers and number of nodes = {3, 2, 2, 2}. It also assumes a
* sigmoid activation function.
*
* @author: James "Andy" Edmond
* @date: April 27, 2020
*/

TEST(ForwardPropTest, UniformWeightsAndBiases_UniformInput) {
  DNN network_test = DNN("this_is_a_bad_path.csv");
  int num_layers = 4;
  int layer_sizes[5] = {3, 2, 2, 2};
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
  answer.fill(0.9579);
  const double tol = 0.01;
  for (int i = 0; i < answer.size(); i++) {
    double diff = abs(answer(i) - output(i));
    ASSERT_TRUE(diff < tol);
  }
}


TEST(ForwardPropTest, NonUniformWeightsAndBiases_UniformInput) {
  DNN network_test = DNN("this_is_a_bad_path.csv");
  int num_layers = 4;
  int layer_sizes[4] = {3, 2, 2, 2};
  // confirm sizes are right
  xt::xtensor<int,1> sizes = network_test.get_layer_sizes();
  for (int i = 0; i < network_test.get_num_layers(); i++) {
    assert( layer_sizes[i] == sizes(i) );
  }

  // set weights and biases to to int of recv_node / 10
  arr weights_1 = {{0.25, 0.5}, {0.5, 0.75}, {0.75, 0.33}};
  arr weights_2 = {{0.25, 0.5}, {0.5, 0.75}};
  arr weights_3 = {{0.33, 0.66}, {0.66, 1.0} };
  arr all_weights[3] = {weights_1, weights_2, weights_3};
  vec biases_1 = {0, 0};
  vec biases_2 = {0, 0};
  vec biases_3 = {0, 0};
  vec all_biases[3] = {biases_1, biases_2, biases_3};
  for (int i = 1; i < num_layers; i++) {
    network_test.set_weights(i,all_weights[i - 1]);
    network_test.set_biases(i,all_biases[i - 1]);
  }

  // make input vec
  vec input = {0.333, 0.666, 1.000};

  // do computations
  network_test.forward_propagate(input);

  // compare output to guess
  vec output = network_test.get_activations(num_layers-1);
  vec answers = {0.6658, 0.7577};
  const double tol = 0.01;
  for (int i = 0; i < answers.size(); i++) {
    double diff = abs(output(i) - answers(i));
    ASSERT_TRUE(diff < tol);
  }
}
