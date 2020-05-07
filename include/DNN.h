#ifndef DNN_header
#define DNN_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xnpy.hpp"
//#include "xtensor-blas/xlinalg.hpp"
#include <istream>
#include <fstream>
#include <iostream>
//#include <vector>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdlib.h>
#include <cstring>
#include <time.h>

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


// Computes avg of xtensor
inline double compute_avg(vec& input_vec) {
  double val = 0;
  for (int i = 0; i < input_vec.size(); i++) {
    val += input_vec(i);
  }
  return val / input_vec.size();
}

// Given a tolerance to compare doubles, see if two double arrs are equal
inline bool arrs_equal(arr& arr1, arr& arr2, double tol) {
  assert( arr1.shape()[0] == arr2.shape()[0] );
  assert( arr1.shape()[1] == arr2.shape()[1] );
  for (int i = 0; i < arr1.shape()[0]; i++) {
    for (int a = 0; a < arr1.shape()[1]; a++) {
      if ( abs( arr1(i,a) - arr2(i,a) ) > tol ) { return false; }
    }
  }
  return true;
}

// Given a tolerance to compare doubles, see if two vecs are equal
inline bool vecs_equal(vec& vec1, vec& vec2, double tol) {
  assert( vec1.size() == vec2.size() );
  for (int i = 0; i <  vec1.size(); i++) {
    if ( abs( vec1(i) - vec2(i) ) > tol ) { return false; }
  }
  return true;
}

// --- uint8_t ==> int coverter ---

// Takes a vector of type uint8_t and returns vec
inline vec make_double_vector(std::vector<uint8_t> vector_in) {
  vec out_vector = xt::zeros<double>({vector_in.size()});
  for (int i = 0; i < vector_in.size(); i++) {
    out_vector(i) = double( vector_in[i] );
  }
  // 0 produced 7
  std::cout << " orig val " << int(vector_in[1]) << " became " << out_vector(1) << "\n";
  return out_vector;
}


// Takes a vector of vectors (each with type uint8_t) and returns a xtensor of
// vecs
inline xt::xtensor<vec,1> make_double_vector(std::vector<std::vector<uint8_t>> vector_in) {
  xt::xtensor<vec,1> out_vector = xt::empty<vec>({vector_in.size()});
  for (int i = 0; i < vector_in.size(); i++) {
    out_vector(i) = make_double_vector(  vector_in[i]  );
  }
  return out_vector;
}

// Normalize vector of vectors (each with type uint8_t) to xtensor of vecs
inline xt::xtensor<vec,1> normalize(std::vector<std::vector<uint8_t>> data_in) {
  xt::xtensor<vec,1> normed_data = xt::empty<vec>({data_in.size()});
  for (int i = 0; i < data_in.size(); i++) {
    vec single_image = xt::zeros<double>({data_in[i].size()});
    for (int a = 0; a < data_in[i].size(); a++) {
      single_image(a) = double(  (data_in[i])[a]  ) / 255; // 255 is max
    }
    normed_data(i) = single_image;
  }
  return normed_data;
}


// --- Declarations ---
// DNN class
class DNN {
  private:
    // class params
    std::string data_folder_path;  // <-- string reps the folder of saved weights/biases i.e. "/home/me/DNN_Data"
    const int NUM_LAYERS = 4;
    // found intersting rule-of-thumb formula for determining num of hidden nodes:
    // num_hidden = sample_size/(a*(num_input + num_output)), 2 <= a <= 10
    const int LAYER_SIZES[5] = {784, 20, 15, 10};
    const double LEARNING_RATE = 1.0;
    const double TOLERANCE = 0.1;
    const int SAVE_NUM = 1000;  // <-- after this many tests, saved weight and bias data to files
    const int MINI_BATCH_SIZE = 30;
    arr all_weights[4]; // dimens = num_layers - 1
    vec all_biases[4];  // dimens = num_layers - 1
    vec all_activations[5]; // dimens = num_layers
    // Non-trivial funcs
    double activ_func(double val_in);
    double activ_func_deriv(double val_in);
    double inv_activ_func(double val_in);
    double cost_func(double guess, double answer);
    double cost_func_deriv(double guess, double answer);
    vec compute_cost_gradient(vec& answer);
    void save_data();
    void initialize_weights(arr& weights);
    void initialize_biases(vec& biases);
    void initialize_network();
  public:
    DNN(std::string path_to_saves);
    // Getters & Setters
    int get_num_layers();
    xt::xtensor<int,1> get_layer_sizes();
    vec& get_activations(int layer_num);
    void set_activations(int layer_num, vec& activ_in);
    vec& get_biases(int layer_num);
    void set_biases(int layer_num, vec& biases_in);
    arr& get_weights(int layer_num);
    void set_weights(int alyer_num, arr& weights_in);
    int get_num_send_nodes(arr& weights);
    int get_num_recv_nodes(arr& weights);
    // Non-trivial functions
    void forward_propagate(vec& input);
    bool analyze_output(vec& answers);
    void backpropagate(vec& answers);
    void train_network(xt::xtensor<vec,1> images_in, vec labels_in);
    void print_all_weights();
};

#endif //
