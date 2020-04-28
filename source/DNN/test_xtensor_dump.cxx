#include "DNN.h"

int main() {

  vec test_me = {0., 1., 2., 3.};
  xt::xarray<double> test_me_too = {0., 1., 2., 3.};

  /**
  * Hmmm... get same error with both...
  * terminate called after throwing an instance of 'std::runtime_error'
  * what():  Only 2-D expressions can be serialized to CSV
  * Aborted (core dumped)
  */

  //std::ofstream out_file("/home/jae1018/Proj2/MNIST_DeepLearning/build/out_test.csv");
  //xt::dump_csv(out_file, test_me_too);

  // Here's their example (modified for ofstream constructor to work)

  std::ofstream out_file("/home/jae1018/Proj2/MNIST_DeepLearning/build/out.csv");
  //std::ofstream out_file("out.csv");

  //xt::xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
  xt::xtensor<double,2> a = {{1,2,3,4}, {5,6,7,8}};
  //xt::dump_csv(out_file, a);
  xt::dump_csv(out_file, test_me);

  return 0;
}
