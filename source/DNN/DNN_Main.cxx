#include "DNN.h"

int main() {
  DNN node_man = DNN();
  std::cout << "node_man made fhgsdgs?\n";
  node_man.print_all_inputs();
  node_man.print_all_weights();
  node_man.compute_output();
}

