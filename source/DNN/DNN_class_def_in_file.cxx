#include "DNN.h"
#include <vector>

class DNN {
  private:

    // ---------- Private Parameters ----------

    vec layers[5];

    // ---------- Private Functions ----------

    // Returns weight of node at layer and node specified
    double get_node_weight(int layer_num, int node_num) {
      return (layers[layer_num])(node_num);
    }

    // Sets weight of node at layer and node specified
    void set_node_weight(int layer_num, int node_num, double new_weight) {
      (layers[layer_num])(node_num) = new_weight;
    }

    // Starts nodes out from scratch (no prior weights saved to file imported here!!)
    void initialize_nodes() {
      for (int i = 0; i < num_layers; i++) {
        vec new_layer = xt::empty<double>({layer_sizes[i]});
        for (int node = 0; node < layer_sizes[i]; node++) {
          new_layer(node) = univ_starting_weight;
        }
        layers[i] = new_layer;
      }
    }

  public:

    // ---------- Public Parameters ----------

    const int num_layers = 5;
    const int layer_sizes[5] = {5,5,5,5,5};
    const double univ_starting_weight = .5;

    // ---------- Public Constructor ----------

    DNN() {
      initialize_nodes();
    }

    // ---------- Public Functions ----------

    void print_all_weights() {
      for (int i = 0; i < num_layers; i++) {
        std::cout << "[ ";
        for (int node = 0; node < layer_sizes[i]; node++) {
          std::cout << get_node_weight(i,node) << " ";
        }
        std::cout << "]\n";
      }
    }

};
// ***************************** putting main here for now ************
int main() {
  DNN node_man = DNN();
  std::cout << "node_man made fhgsdgs?\n";
  node_man.print_all_weights();
}
