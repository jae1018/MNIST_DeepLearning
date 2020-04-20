#ifndef Node_header
#define Node_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <iostream>
//#include "this_is_bad.h"
//#include "DNN.h"

// Define vec to replace xtensor<double,1>
using vec = xt::xtensor<double,1>;
//using all_zeros = xt::empty<double>;  <-- doesn't work for some reason???

// Declarations

// Node class
class Node {
  private:
    double input;
    vec weights;
  public:
    Node(double input_in, vec weights_in);
    double get_input();
    double get_weight(int i);
    void set_input();
    void set_weight();
}; 


#endif //
