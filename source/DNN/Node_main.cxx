#include "Node.h"

int main() {
  double val = 5.3;
  vec arr = {1., 2., 3.};
  Node test_me = Node(val,arr);
  std::cout << "Node has val = " << test_me.get_input() << "\n";
  std::cout << "Node has weights: [";
  for (int i = 0; i < arr.size(); i++) {
    std::cout << " " << test_me.get_weight(i);
  }
  std::cout << " ]\n";
}
