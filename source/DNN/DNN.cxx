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


// Returns biases for designated layer
// Note that input layer has no biases, so get_biases(0) throws error!
vec& DNN::get_biases(int layer_num) {
  return all_biases[layer_num-1];
}


// ----- Non-trivial Functions -----

/**
// The activation function for the neural network
double DNN::activ_func(double val)  {
  return 1/(1 + exp(-val));
}


// Calculates the preactivation value for a node (the linear combination
// that needs to be performed i.e. x1*w1 + x2*w2 + ...)
// Note that node_layer describes the layer data is going *to* and
// node_int describes the node eventually receiving that data.
// So all data comes from layer node_layer - 1
double DNN::calc_preactiv(int node_layer,int node_int) {
  double val = 0;
  std::cout << get_all_weights(node_layer,node_int) << " *****\n";
  std::cout << get_weight(node_layer,9,node_int) << " *****\n";
  for (int i = 0; i < LAYER_SIZES[node_layer-1]; i++) {
    val += get_input(node_layer-1,i) * get_weight(node_layer,i,node_int);
  }
  //val += get_input(node_layer-1,-1) * get_weight(node_layer,-1,node_int);
  return val;
}


// Forward propogates data from input layer all the way to output layer
void DNN::forward_propogate() {
  for (int layer_num = 1; layer_num < NUM_LAYERS; layer_num++) {
    for (int node_num = 0; node_num < LAYER_SIZES[layer_num]; node_num++) {
      double preactiv_val = calc_preactiv(layer_num,node_num);
      //set_input(layer_num, node_num, activ_func(preactiv_val));
    }
  }
}*/


// Fill up all entries in weights arr
void DNN::fill_up_weights(arr& weights) {
  int num_recv = get_num_recv_nodes(weights);
  int num_send = get_num_send_nodes(weights);
  for (int recv = 0; recv < num_recv; recv++) {
    for (int send = 0; send < num_send; send++) {
      weights(send,recv) = double(recv*10 + send)/100;
    }
  }
}


// Fill up all entires in biases vec
void DNN::fill_up_biases(vec& biases) {
  for (int i = 0; i < biases.size(); i++) {
    biases(i) = double(i)/100;
  }
}


// Starts nodes out from scratch (no prior weights saved to file imported here!!)
void DNN::initialize_network() {
  for (int i = 0; i < NUM_LAYERS-1; i++) {
    // Make weights matrix between adjacent layers
    arr weights = xt::empty<double>({LAYER_SIZES[i],LAYER_SIZES[i+1]});
    fill_up_weights(weights);
    all_weights[i] = weights;
    // Make biases for non-input layers
    vec biases = xt::empty<double>({LAYER_SIZES[i+1]});
    fill_up_biases(biases);
    all_biases[i] = biases;
  }
}


// ----- Public Constructor -----


DNN::DNN() {
  initialize_network();
}


// ----- Public Functions -----


// Prints all the weights for all nodes
void DNN::print_all_weights() {
  std::cout << "----- printing weights -----\n"
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
/**void DNN::compute_output() {
  std::cout << "----- forward propagating -----\n";
  forward_propogate();
  std::cout << "Output is " << get_input(NUM_LAYERS - 1, 0) << "\n";
}*/
