#ifndef DNN_header
#define DNN_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <iostream>
//#include <vector>
#include <cmath>

// Define vec to replace xtensor<double,1>
using vec = xt::xtensor<double,1>;
using arr = xt::xtensor<double,2>;
//using all_zeros = xt::empty<double>;  <-- doesn't work for some reason???

// --- Declarations ---
// Node class
/**class Node {
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
};*/

// DNN class
class DNN {
  private:
    const int NUM_LAYERS = 5;
    const int LAYER_SIZES[5] = {10, 8, 6, 4, 2};
    arr all_weights[4]; // dimens = num_layers - 1
    vec all_biases[4];  // dimens = num_layers - 1
    // Getters & Setters
    vec& get_biases(int layer_num);
    arr& get_weights(int layer_num);
    int get_num_send_nodes(arr& weights);
    int get_num_recv_nodes(arr& weights);
    /**void set_input(int layer_num, int node_num, double new_input);
    double get_input(int layer_num, int node_num);
    double get_weight(int layer_num, int node_num, int weight_num);
    xtens get_all_weights(int layer_num, int node_num);
    void set_node_weight(int layer_num, int node_num, double new_weight);
    */
    // Non-trivial funcs
    //double activ_func(double val_in);
    //double calc_preactiv(int layer_num, int node_num);
    //void forward_propogate();
    void fill_up_weights(arr& weights);
    void fill_up_biases(vec& biases);
    void initialize_network();
  public:
    DNN();
    //void print_all_inputs();
    void print_all_weights();
    //void compute_output();
};

#endif //
