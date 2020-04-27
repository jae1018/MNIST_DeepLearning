#ifndef DNN_header
#define DNN_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
//#include "xtensor-blas/xlinalg.hpp"
#include <iostream>
//#include <vector>
#include <cmath>
#include <cassert>
#include <vector>

// Define vec to replace xtensor<double,1>
using vec = xt::xtensor<double,1>;
using arr = xt::xtensor<double,2>;
//using vdot = xt::linalg::vdot;  <-- use for vectors with (a,b)
//using dot = xt::linalg::dot  <-- use for matrix vector mult with (a[matrix],b[vec])

// --- LinAlg Definitions ---

// Get multiple def error when including this header across multiple files and its due to this...
// can make it inline possibly to avoid that?
inline double vdot(vec& vec_one, vec& vec_two) {
  double val = 0;
  assert(vec_one.size() == vec_two.size());
  for (int i = 0; i < vec_one.size(); i++) {
    val += vec_one(i) * vec_two(i);
  }
  return val;
}

// --- uint8_t ==> int coverter ---

// Takes a vector of type uint8_t and returns a vector of type int
/**inline std::vector<int> make_int_vector(std::vector<uint8_t> vector_in) {
  std::vector<int> int_vector(vector_in.size(),0);
  for (int i = 0; i < vector_in.size(); i++) {
    int_vector[i] = int( vector_in[i] );
  }
  return int_vector;
}*/

inline vec make_double_vector(std::vector<uint8_t> vector_in) {
  vec out_vector = xt::zeros<double>({vector_in.size()});
  for (int i = 0; i < vector_in.size(); i++) {
    out_vector(i) = double( vector_in[i] );
  }
  return out_vector;
}


// Takes a vector of vectors (each with type uint8_t) and returns a vector of
// vectors (each with type int)
/**inline std::vector<std::vector<int>> make_int_vector(std::vector<std::vector<uint8_t>> vector_in) {
  std::vector<std::vector<int>> int_vector(vector_in.size(),std::vector<int>());
  for (int i = 0; i < vector_in.size(); i++) {
    int_vector[i] = make_int_vector(  vector_in[i]  );
  }
  return int_vector;
}*/

inline xt::xtensor<vec,1> make_double_vector(std::vector<std::vector<uint8_t>> vector_in) {
  xt::xtensor<vec,1> out_vector = xt::empty<vec>({vector_in.size()});
  for (int i = 0; i < vector_in.size(); i++) {
    out_vector(i) = make_double_vector(  vector_in[i]  );
  }
  return out_vector;
}


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
    //const int LAYER_SIZES[5] = {20, 8, 6, 4, 2};
    const int LAYER_SIZES[5] = {784, 100, 60, 30, 10};
    const double LEARNING_RATE = 0.05;
    arr all_weights[4]; // dimens = num_layers - 1
    vec all_biases[4];  // dimens = num_layers - 1
    vec all_activations[5]; // dimens = num_layers
    // Getters & Setters
    vec& get_activations(int layer_num);
    void set_activations(int layer_num, vec& activ_in);
    vec& get_biases(int layer_num);
    void set_biases(int layer_num, vec& biases_in);
    arr& get_weights(int layer_num);
    void set_weights(int alyer_num, arr& weights_in);
    int get_num_send_nodes(arr& weights);
    int get_num_recv_nodes(arr& weights);
    // Non-trivial funcs
    double activ_func(double val_in);
    double activ_func_deriv(double val_in);
    double inv_activ_func(double val_in);
    double cost_func(double guess, double answer);
    double cost_func_deriv(double guess, double answer);
    void forward_propagate();
    void backpropagate(vec& answers);
    void initialize_weights(arr& weights);
    void initialize_biases(vec& biases);
    void initialize_network();
  public:
    DNN();
    //void print_all_inputs();
    void train_network(xt::xtensor<vec,1> images_in, vec labels_in);
    void print_all_weights();
    void compute_forward();
    void compute_backward();
};

#endif //
