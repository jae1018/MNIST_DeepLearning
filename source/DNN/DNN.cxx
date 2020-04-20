#include "DNN.h"
#include <vector>

    // Returns weight of node at layer and node specified
    double DNN::get_node_weight(int layer_num, int node_num) {
      return (layers[layer_num])(node_num);
    }

    // Sets weight of node at layer and node specified
    void DNN::set_node_weight(int layer_num, int node_num, double new_weight) {
      (layers[layer_num])(node_num) = new_weight;
    }

    // Starts nodes out from scratch (no prior weights saved to file imported here!!)
    void DNN::initialize_nodes() {
      for (int i = 0; i < num_layers; i++) {
        vec new_layer = xt::empty<double>({layer_sizes[i]});
        for (int node = 0; node < layer_sizes[i]; node++) {
          new_layer(node) = univ_starting_weight;
        }
        layers[i] = new_layer;
      }
    }

    DNN::DNN() {
      initialize_nodes();
    }

    void DNN::print_all_weights() {
      for (int i = 0; i < num_layers; i++) {
        std::cout << "[ ";
        for (int node = 0; node < layer_sizes[i]; node++) {
          std::cout << get_node_weight(i,node) << " ";
        }
        std::cout << "]\n";
      }
    }
