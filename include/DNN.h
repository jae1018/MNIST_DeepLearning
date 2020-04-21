#ifndef DNN_header
#define DNN_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Define vec to replace xtensor<double,1>
using xtens = xt::xtensor<double,1>;
//using all_zeros = xt::empty<double>;  <-- doesn't work for some reason???

// --- Declarations ---
// Node class
class Node {
  private:
    double input;
    xtens weights;
  public:
    Node(double input_in, xtens weights_in);
    double get_input();
    double get_weight(int i);
    xtens get_all_weights();
    void set_input(double new_input);
    void set_weight(int index, double new_weight);
}; 

// DNN class
class DNN {
  private:
    std::vector<Node> layers[5];
    std::vector<Node> biases;
    // Getters & Setters
    void set_input(int layer_num, int node_num, double new_input);
    double get_input(int layer_num, int node_num);
    double get_weight(int layer_num, int node_num, int weight_num);
    xtens get_all_weights(int layer_num, int node_num);
    //void set_node_weight(int layer_num, int node_num, double new_weight);
    // Non-trivial funcs
    double activ_func(double val_in);
    double calc_preactiv(int layer_num, int node_num);
    void forward_propogate();
    void initialize_network();
  public:
    const int num_layers = 5;
    const int layer_sizes[5] = {10,8,6,4,2};
    const double univ_starting_weight = .5;
    DNN();
    void print_all_inputs();
    void print_all_weights();
    void compute_output();
};

#endif //
