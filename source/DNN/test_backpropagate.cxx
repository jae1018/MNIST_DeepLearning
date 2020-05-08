#include <gtest/gtest.h>
#include "DNN.h"

/**
* Tests the backward propagation of the DNN assuming the biases are 0,
* the weights are what are specified below, and the learning rate = 1.0
* Requires 4 layers with the structure {3, 2, 2, 2}, that the activation
* function is sigmoid, and that the cost function is quadratic.
*
* @author: James "Andy" Edmond
* @date: April 27, 2020
*/


// Tests backpropagation for the setup in the first forward propagation test.
// PRESUMES THAT FORWARD_PROPAGATION WORKS and that DNN::LEARNING_RATE = 0.05
TEST(BackPropTest, UniformWeightsAndBiases_UniformInput) {
  DNN network_test = DNN("this_is_a_bad_path.csv");
  int num_layers = 4;
  int layer_sizes[5] =  {3, 2, 2, 2};
  // confirm sizes are right
  xt::xtensor<int,1> sizes = network_test.get_layer_sizes();
  for (int i = 0; i < network_test.get_num_layers(); i++) {
    assert( layer_sizes[i] == sizes(i) );
  }

  // set weights and biases to uniform

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
  vec last_activs = network_test.get_activations(num_layers - 1);
  vec last_activs_deriv = last_activs * (1 - last_activs);
  //std::cout << "output = " << last_activs << "\n";
  vec test_output = {0.0, 1.0};
  vec cost_gradient = last_activs - test_output;
  //std::cout << "cost grad expected = " << cost_gradient << "\n";
  //std::cout << "final layer error expected = " << last_activs_deriv * cost_gradient << "\n";
  network_test.backpropagate(cost_gradient);

  // Create predictions
  arr weights_one = { {0.2496, 0.4993}, {0.4992, 0.7486}, {0.7488, 0.3309} };
  arr weights_two = { {0.2465, 0.4916}, {0.4967, 0.7420} };
  arr weights_three = { {0.2389, 0.6943}, {0.5598, 1.0319} };
  arr all_correct_weights[3] = { weights_one, weights_two, weights_three };
  /**vec biases_one = xt::zeros<double>({layer_sizes[1]});
  biases_one.fill(0.9999999962);
  vec biases_two = xt::zeros<double>({layer_sizes[2]});
  biases_two.fill(0.9999995);
  vec biases_three = xt::zeros<double>({layer_sizes[3]});
  biases_three.fill(0.9999621);
  vec biases_four = {0.9977797, 1.00011465};
  vec all_correct_biases[4] = { biases_one, biases_two, biases_three, biases_four };*/

  // Then get weights and compare to expected weights
  const double tol = 0.01;
  for (int i = num_layers - 1; i > 0; i--) {
    arr computed_weights = network_test.get_weights(i);
    arr correct_weights = all_correct_weights[i - 1];
    //std::cout << "For layer " << i + 1 << "...\n"
    //          << "correct weights " << correct_weights << "\n"
    //          << "calculated weights " << computed_weights << "\n";
    ASSERT_TRUE( arrs_equal(computed_weights, correct_weights, tol) );
    /**vec computed_biases = network_test.get_biases(i);
    vec correct_biases = all_correct_biases[i - 1];
    ASSERT_TRUE( vecs_equal(computed_biases, correct_biases, tol) );*/
  }
}
