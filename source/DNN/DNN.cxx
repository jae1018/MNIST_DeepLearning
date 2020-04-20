#include "DNN.h"
#include <vector>

// ********** Private Functions **********

// ----- Getters & Setters -----

// Sets input for node at layer and node specified
void DNN::set_input(int layer_num, int node_num, double new_input) {
  (  (  layers[layer_num]  ).at(node_num)  ).set_input(new_input);
}

// Returns input of node at layer and node specified
double DNN::get_input(int layer_num, int node_num) {
  return (  (  layers[layer_num]  ).at(node_num)  ).get_input();
}

// Returns weight for node connection at layer, node, and connec specified.
double DNN::get_weight(int layer_num, int node_num, int weight_num) {
  return (  (  layers[layer_num]  ).at(node_num)  ).get_weight(weight_num);
}

// Returns *all* weights that a particular node maintains.
xtens DNN::get_all_weights(int layer_num, int node_num) {
  return (  (  layers[layer_num]  ).at(node_num)  ).get_all_weights();
}

/**   * not funcitonal yet!!! *
// Sets weight of node at layer and node specified
void DNN::set_node_weight(int layer_num, int node_num, double new_weight) {
  (  (  layers[layer_num]  ).at(node_num)  ).set_weight = new_weight;
}*/

// ----- Non-trivial Functions -----

// The activation function for the neural network
double DNN::activ_func(double val)  {
  return val;  // %%%%%  fix me!!! %%%%%
}

// Calculates the preactivation value for a node (the linear combination
// that needs to be performed i.e. x1*w1 + x2*w2 + ...)
// Note that data_layer describes the layer we're taking data *from* and
// node_int describes the node eventually *receiving* the data in the
// **next** layer! (so if data_layer = 0, then node_int is in layer = 1)
double DNN::calc_preactiv(int data_layer,int node_int) {
  double val = 0;
  for (int i = 0; i < layer_sizes[data_layer]; i++) {
    val += get_input(data_layer,i) * get_weight(data_layer,i,node_int);
  }
  return val;
}

// Forward propogates data from input layer all the way to output layer
void DNN::forward_propogate() {
  for (int layer_num = 1; layer_num < num_layers; layer_num++) {
    for (int node_num = 0; node_num < layer_sizes[layer_num]; node_num++) {
      double preactiv_val = calc_preactiv(layer_num - 1,node_num);
      set_input(layer_num, node_num, activ_func(preactiv_val));
    }
  }
}

// Starts nodes out from scratch (no prior weights saved to file imported here!!)
void DNN::initialize_nodes() {
  // For each layer ...
  for (int i = 0; i < num_layers; i++) {
    // Make a node ...
    for (int num_node = 0; num_node < layer_sizes[i]; num_node++)  {
      // With weights for each connection to node in next layer ...
      xtens weights;
      if (i < (num_layers - 1)) {  // no weights needed for output layer
	weights = xt::empty<double>({layer_sizes[i+1]});
        weights.fill(univ_starting_weight);  // try RNG next time
      }
      // Then create the node and add it to the vector
      Node new_node = Node(0.5, weights);
      (layers[i]).push_back(new_node);
    }
  }
}

// ----- Public Constructor -----

DNN::DNN() {
  initialize_nodes();
}

// ----- Public Functions -----

// Prints the input for all nodes
void DNN::print_all_inputs() {
  std::cout << "----- printing inputs -----\n";
  for (int i = 0; i < num_layers; i++) {
    std::cout << "*** Layer " << i + 1 << " ***\n";
    std::cout << "[ ";
    for (int node_num = 0; node_num < layer_sizes[i]; node_num++) {
      std::cout << get_input(i,node_num) << " ";
    }
    std::cout << "]\n\n";
  }
}

// Prints all the weights for all nodes
void DNN::print_all_weights() {
  std::cout << "----- printing weights -----\n";
  for (int i = 0; i < num_layers; i++) {
    std::cout << "*** Layer " << i + 1 << " ***\n";
    for (int node_num = 0; node_num < layer_sizes[i]; node_num++) {
      std::cout << "Node " << node_num + 1 << " has the following weights:\n";
      xtens weights = get_all_weights(i,node_num);
      std::cout << "[ ";
      for (int weight_num = 0; weight_num < weights.size(); weight_num++) {
        std::cout << weights(weight_num) << " ";
      }
      std::cout << "]\n";
    }
    std::cout << "\n";
  }
}

// Test of forward propogate
void DNN::compute_output() {
  std::cout << "----- forward propagating -----\n";
  forward_propogate();
  std::cout << "Output is " << get_input(num_layers - 1,0) << "\n";
}
