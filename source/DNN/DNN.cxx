#include "DNN.h"

/**
* Note: Bias nodes are modified by setting a node_int val
* of -1 to the set_input and set_weight functions.
*
*
*
* STRUCTURE OF NETWORK:
*
* *** Input Layer (0) ***          *** Layer 1 ***
* ----------------------       -----------------------|
* | input has  | input |       | layer 1 has| Layer 1 |
* | no weights | nodes |       | weights    | Nodes   |
* |            |       | ----\ | for nodes  |         | -----\ ... until final
* |------------|-------| ----/ | from layer |         | -----/ ... layer.
* | and no bias nodes  |       | 0 to layer |         |
* | either             |       | 1 (called  |         |
* ---------------------        | w^(L=1)    |         |
*                              ------------------------
*                              | biases for each node |
*                              | activation in layer 1|
*                              | (called b^(L=1))     |
*                              ------------------------
*
* @author: James "Andy" Edmond
* @date: April 20, 2020 (420lol)
*/

// ********** Private Functions **********


// ----- Getters & Setters -----


// Based on ref to weight matrix, get number of receiving nodes
int DNN::get_num_recv_nodes(arr& weights) {
  return (  weights.shape()  )[1];
}


// Based on ref to weight matrix, get number of sending nodes
int DNN::get_num_send_nodes(arr& weights) {
  return (  weights.shape()  )[0];
}


// Returns weights for designated layer
// Note that input layer has no weights, so get_weights(0) throws error!
arr& DNN::get_weights(int layer_num) {
  return all_weights[layer_num-1];
}


// Sets the weights of the designated layer to the arr provided
void DNN::set_weights(int layer_num, arr& weights_in) {
  double rows = ((all_weights[layer_num - 1]).shape())[0];
  double cols = ((all_weights[layer_num - 1]).shape())[1];
  assert(  rows == (weights_in.shape())[0]  );
  assert(  cols == (weights_in.shape())[1]  );
  for (int i = 0; i < rows; i++) {
    for (int a = 0; a < cols; a++) {
      (all_weights[layer_num - 1])(i,a) = weights_in(i,a);
    }
  }
}


// Returns biases for designated layer
// Note that input layer has no biases, so get_biases(0) throws error!
vec& DNN::get_biases(int layer_num) {
  return all_biases[layer_num - 1];
}


// Sets the biases of the designated layer to the vec provided
void DNN::set_biases(int layer_num, vec& biases_in) {
  assert( (all_biases[layer_num - 1]).size() == biases_in.size() );
  for (int i = 0; i < biases_in.size(); i++) {
    (all_biases[layer_num - 1])(i) = biases_in(i);
  }
}


// Returns activations for designated layer
// The input values for the input layer are considered to be activations!
vec& DNN::get_activations(int layer_num) {
  return all_activations[layer_num];
}


// Sets the activations of the designated layer to the vec provided
void DNN::set_activations(int layer_num, vec& activ_in) {
  assert((all_activations[layer_num]).size() == activ_in.size());
  for (int i = 0; i < activ_in.size(); i++) {
    (all_activations[layer_num])(i) = activ_in(i);
  }
}


// ----- Non-trivial Functions -----


// The activation function for the neural network
double DNN::activ_func(double val)  {
  return 1/(1 + exp(-val));
}


// The derivative of the activation function
double DNN::activ_func_deriv(double val) {
  return 1/(2 * (1 + cosh(val)) );
}


// Inverse of the activation function
// (Needed for going from activation to weighted sum)
double DNN::inv_activ_func(double val) {
  return -1 * log( (1/val) - 1 );
}


// The cost function used for evaluating error
// Note that technically, the cost function is evaluated over the sum of vector elements! But
// the result defined below is desired for computation.
double DNN::cost_func(double guess, double answer) {
  return pow((answer - guess),2) / 2;
}


// Derivative of the cost function
// (Change this when modifying cost function!)
double DNN::cost_func_deriv(double guess, double answer) {
  return (guess - answer);
}


// Forward propogates data from input layer all the way to output layer
void DNN::forward_propogate() {
  for (int layer_num = 1; layer_num < NUM_LAYERS; layer_num++) {

    // get data for calculations
    vec prev_activ_data = get_activations(layer_num - 1);  // activ data for layer L - 1
    vec next_activ_data = get_activations(layer_num);  // activ data for layer L
    arr weights = get_weights(layer_num);  // weights between layers L - 1 and L
    vec biases = get_biases(layer_num);  // biases for layer L
    int num_recv = get_num_recv_nodes(weights);

    // forward propagate to next layer
    for (int recv_node = 0; recv_node < num_recv; recv_node++) {
      vec weights_slice = xt::col(weights,recv_node);
      double weighted_sum = vdot(weights_slice,prev_activ_data) + biases(recv_node);
      next_activ_data(recv_node) = activ_func(weighted_sum);
    }
    set_activations(layer_num,next_activ_data);

  }
}


// Backward propagates based on correct_answer vec (which contains the correct answer to the
// current trial). Starts by computing error associated with cost function for final layer
// and propagates the error backward. After all error is determined, weights and biases
// are updated.
void DNN::backpropagate(vec& answers) {

  // Compute error of final layer
  std::cout << "*** Computing error for last layer ***\n";
  std::cout << "From answers = " << answers << " errors are determined\n";
  int last_layer_index = NUM_LAYERS - 1;
  vec last_activs = get_activations(last_layer_index);
  vec error_vec = xt::zeros<double>({LAYER_SIZES[last_layer_index]});
  for (int i = 0; i < LAYER_SIZES[last_layer_index]; i++) {
    double from_cost_deriv = cost_func_deriv(last_activs(i),answers(i));
    double from_activ_func_deriv = activ_func_deriv(inv_activ_func(last_activs(i)));
    error_vec(i) = from_cost_deriv * from_activ_func_deriv;
    std::cout << "Node " << i << " has error " << error_vec(i) << "\n";
  }

  // Start saving errors for updating later
  // (only 4 vecs needed since no weights on input layer)
  vec errors[NUM_LAYERS - 1];
  errors[last_layer_index - 1] = error_vec;

  // Compute errors from semi-final layer to input layer
  for (int i = last_layer_index - 1; i > 0; i--) {
    std::cout << "\n*** Computing error for layer " << i + 1 << " ***\n";
    //vec prev_errors = errors[i];
    arr weights = get_weights(i + 1);
    int num_send_nodes = get_num_send_nodes(weights);
    vec upper_layer_activs = get_activations(i);
    vec lower_layer_errors = xt::zeros<double>({num_send_nodes});
    vec upper_layer_errors = errors[i];
    std::cout << "upper_layer_errors = " << upper_layer_errors << "\n";
    // Compute error for individual neuron
    for (int q = 0; q < num_send_nodes; q++) {
      vec weight_slice = xt::row(weights,q);
      double weight_and_error_dot = vdot(weight_slice,upper_layer_errors);
      double from_cost_func_deriv = activ_func_deriv(  inv_activ_func(  upper_layer_activs(q)  )  );
      lower_layer_errors(q) = weight_and_error_dot * from_cost_func_deriv;
      std::cout << "Node " << q << " has error " << lower_layer_errors(q) << "\n";
    }
    errors[i - 1] = lower_layer_errors;
  }

  // Update weights and biases
  // across each layer ...
  for (int i = 1; i < NUM_LAYERS; i++) {
    arr old_weights = get_weights(i);
    arr new_weights = xt::zeros<double>({ (old_weights.shape())[0], (old_weights.shape())[1] });
    vec old_biases = get_biases(i);
    vec new_biases = xt::zeros<double>({old_biases.size()});
    vec prev_layer_activs = get_activations(i - 1);
    vec errors_for_layer = errors[i - 1]; // --> means errors for layer i
    // ... and across each receiving node ...
    for (int recv = 0; recv < get_num_recv_nodes(old_weights); recv++) {
      // ... update the weights for each connection
      for (int send = 0; send < get_num_send_nodes(old_weights); send++) {
        double change = LEARNING_RATE * errors_for_layer(recv) * prev_layer_activs(send);
        // should it be plus???
        new_weights(send,recv) = old_weights(send,recv) - change;
      }
      // ... and update the bias for each receiving node
      new_biases(recv) = old_biases(recv) - LEARNING_RATE * errors_for_layer(recv);
    }
    set_weights(i,new_weights);
    set_biases(i,new_biases);
  }

}


// Fill up all entries in weights arr
void DNN::initialize_weights(arr& weights) {
  int num_recv = get_num_recv_nodes(weights);
  int num_send = get_num_send_nodes(weights);
  for (int recv = 0; recv < num_recv; recv++) {
    for (int send = 0; send < num_send; send++) {
      weights(send,recv) = double(recv*10 + send)/100;
    }
  }
}


// Fill up all entires in biases vec
void DNN::initialize_biases(vec& biases) {
  for (int i = 0; i < biases.size(); i++) {
    biases(i) = double(i)/100;
  }
}


// Starts nodes out from scratch (no prior weights saved to file imported here!!)
void DNN::initialize_network() {
  // Init weights and biases (incorpoate file-checking for prev vals later!)
  for (int i = 0; i < NUM_LAYERS-1; i++) {
    // Make weights matrix between adjacent layers
    arr weights = xt::empty<double>({LAYER_SIZES[i],LAYER_SIZES[i+1]});
    initialize_weights(weights);
    all_weights[i] = weights;
    // Make biases for non-input layers
    vec biases = xt::empty<double>({LAYER_SIZES[i+1]});
    initialize_biases(biases);
    all_biases[i] = biases;
  }
  // Init activations
  for (int i = 0; i < NUM_LAYERS; i++) {
    vec activations = xt::zeros<double>({LAYER_SIZES[i]});
    all_activations[i] = activations;
  }
}


// In the future, this will take data from a file or maybe a prepared arr
// For now, it just inits the input layer to non-zero stuff
void DNN::read_input() {
  for (int i = 0; i < (all_activations[0]).size(); i++) {
    (all_activations[0])(i) = double(i)/10;
  }
}

// ----- Public Constructor -----


DNN::DNN() {
  initialize_network();
}


// ----- Public Functions -----


// Prints all the weights for all nodes
void DNN::print_all_weights() {
  std::cout << "\n----- printing weights -----\n"
            << "*** Layer 1 is input layer (so no weights or biases) ***\n";
  for (int i = 1; i < NUM_LAYERS; i++) {
    std::cout << "\n*** Layer " << i + 1 << " ***\n";
    arr weights = get_weights(i);
    vec biases = get_biases(i);
    int num_recv = get_num_recv_nodes(weights);
    int num_send = get_num_send_nodes(weights);
    for (int r_node = 0; r_node < num_recv; r_node++) { // r = receiving
      std::cout << "Node " << r_node + 1 << " has the following weights:\n[ ";
      for (int s_node = 0; s_node < num_send; s_node++) { // s = sending
        std::cout << weights(s_node,r_node) << " ";
      }
      std::cout << "] with bias = " << biases(r_node) << "\n";
    }
  }
}


// Test of forward propogate
void DNN::compute_forward() {
  read_input();
  std::cout << "\n----- forward propagating -----\n";
  forward_propogate();
  std::cout << "Output is " << get_activations(NUM_LAYERS - 1) << "\n";
}


// Test of backpropagate
void DNN::compute_backward() {
  vec answers = {0., 1.};
  std::cout << "\n----- backpropagating -----\n";
  backpropagate(answers);
}
