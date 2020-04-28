#include <gtest/gtest.h>
#include "DNN.h"

/**
* Tests the backward propagation of the DNN assuming the uniform
* weights, biases and input (all = 1) of the first test in the suite
* of forward propagation tests. I'd do more but this stuff is incredily
* meticulous to do be by hand, even for this simple example...
*
* This test suite requires the DNN to be structured with 5
* layers and number of nodes = {5, 4, 3, 2, 2}. Also assumed is that
* the activation function is sigmoid, the cost function is quadratic.
*
* @author: James "Andy" Edmond
* @date: April 27, 2020
*/


// Tests backpropagation for the setup in the first forward propagation test.
// PRESUMES THAT FORWARD_PROPAGATION WORKS and that DNN::LEARNING_RATE = 0.05
TEST(BackPropTest, UniformWeightsAndBiases_UniformInput) {
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
  vec test_output = xt::zeros<double>({layer_sizes[num_layers - 1]});
  test_output(0) = 0;
  test_output(1) = 1;
  network_test.backpropagate(test_output);

  // Create predictions
  arr weights_one = xt::zeros<double>({layer_sizes[0],layer_sizes[1]});//{ {0., 0., 0., 0.}, {0., 0., 0., 0.}, {0., 0., 0., 0.}, {0., 0., 0., 0.}, {0., 0., 0., 0.} };
  weights_one.fill(0.9999999962);
  arr weights_two = xt::zeros<double>({layer_sizes[1],layer_sizes[2]});//{ {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.} };
  weights_two.fill(0.9999995012);
  arr weights_three = xt::zeros<double>({layer_sizes[2],layer_sizes[3]});//{ {0., 0.}, {0., 0.}, {0., 0.} };
  weights_three.fill(0.9999623562);
  arr weights_four = { {0.9978204425, 1.000112546}, {0.9978204425, 1.000112546} };
  arr all_correct_weights[4] = { weights_one, weights_two, weights_three, weights_four};
  vec biases_one = xt::zeros<double>({layer_sizes[1]});
  biases_one.fill(0.9999999962);
  vec biases_two = xt::zeros<double>({layer_sizes[2]});
  biases_two.fill(0.9999995);
  vec biases_three = xt::zeros<double>({layer_sizes[3]});
  biases_three.fill(0.9999621);
  vec biases_four = {0.9977797, 1.00011465};
  vec all_correct_biases[4] = { biases_one, biases_two, biases_three, biases_four };

  // Then get weights and compare to expected weights
  const double tol = 0.0001;
  for (int i = num_layers - 1; i > 0; i--) {
    arr computed_weights = network_test.get_weights(i);
    arr correct_weights = all_correct_weights[i - 1];
    ASSERT_TRUE( arrs_equal(computed_weights, correct_weights, tol) );
    vec computed_biases = network_test.get_biases(i);
    vec correct_biases = all_correct_biases[i - 1];
    ASSERT_TRUE( vecs_equal(computed_biases, correct_biases, tol) );
  }
}
