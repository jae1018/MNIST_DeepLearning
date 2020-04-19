#ifndef DNN_header
#define DNN_header

#include <xtensor/xarray.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <iostream>

// Define vec to replace xtensor<double,1>
using vec = xt::xtensor<double,1>;
//using all_zeros = xt::empty<double>;  <-- doesn't work for some reason???

// Functions declarations;


#endif //
