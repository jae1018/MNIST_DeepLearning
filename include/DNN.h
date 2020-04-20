#ifndef DNN_header
#define DNN_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <iostream>

// Define vec to replace xtensor<double,1>
using vec = xt::xtensor<double,1>;
//using all_zeros = xt::empty<double>;  <-- doesn't work for some reason???

// --- Declarations ---
// Node class
class Node {
  private:
    double input;
    vec weights;
  public:
    Node(double input_in, vec weights_in);
    double get_input();
    double get_weight(int i);
    void set_input(double new_input);
    void set_weight(int index, double new_weight);
}; 

// DNN class
class DNN {
  private:
    vec layers[5];
    double get_node_weight(int layer_num, int node_num);
    void set_node_weight(int layer_num, int node_num, double new_weight);
    void initialize_nodes();
  public:
    const int num_layers = 5;
    const int layer_sizes[5] = {5,5,5,5,5};
    const double univ_starting_weight = .5;
    DNN();
    void print_all_weights();
};

#endif //
