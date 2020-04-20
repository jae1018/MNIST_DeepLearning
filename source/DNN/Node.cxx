//#include "Node.h"
#include "DNN.h"

/**
* Structure for the Node class. Only two class parameters (both private) with
* getters and setters for both. Input is the value stored in the node itself
* and weights is a 1D xtensor that stores the values of weights for
* connections between this node and connected nodes.
* Important distinction here is that the size of weights is not changed after
* initialization! Only inidivual elemtents are modified at a time, and not
* the entire xtensor.
*
* date: April 19, 2020
* author: James "Andy" Edmond
*/

// Constructor
Node::Node(double input_in, xtens weights_in) {
  input = input_in;
  weights = weights_in;
}

// Get input of Node
double Node::get_input() {
  return input;
}

// Get weight for ith connection
double Node::get_weight(int i) {
  return weights(i);
}

// Get all weights that a Node maintains
xtens Node::get_all_weights() {
  return weights;
}

// Changes class param input to that provided
void Node::set_input(double input_in) {
  input = input_in;
}

// Changes *element* at index i of weights to that provided
void Node::set_weight(int i, double new_weight) {
  weights(i) = new_weight;
}
