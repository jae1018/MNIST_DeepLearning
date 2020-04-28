#include <iostream>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "DNN.h"

int main() {

  DNN node_man = DNN("/home/jae1018/Proj2/MNIST_DeepLearning/build");

  // --- Data Reader stuff ---

  // MNIST_DATA_LOCATION set by MNIST cmake config
  std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

  // Loads MNIST data
  // The original author uses a template:
  // template <template <typename...> class Container, typename Image, typename Label>
  // To clarify, the author maintains two different pairs of sets, each with an image
  // set and the other with a label set.
  // *** (1) ***
  // Each element of the IMAGE vector is a vector describing a 28 x 28 pixel image (where
  // each element of the latter vector is a uint8_t [an unsigned char] that describes a
  // corresponding pixel in the image in grayscale format [goes between 0 and 255].
  // *** (2) ***
  // Each element of the LABEL vector is just a uint8_t that goes between 0 and 9. It is the
  // label for each corresponding image and describes the number that the image depicts.
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

  //std::vector<std::vector<int>> images = make_int_vector(dataset.test_images);
  //std::vector<int> labels = make_int_vector(dataset.test_labels);

  // Test Size --> 10K     &&&     Training Size --> 60K
  auto images_orig = dataset.training_images; //dataset.test_images;
  auto labels_orig = dataset.training_labels;  //dataset.test_labels;

  xt::xtensor<vec,1> images = normalize(images_orig);
  vec labels = make_double_vector(labels_orig);

  std::cout << "***Each image has " << images(0).size() << " number of pixels." << std::endl;

  node_man.train_network(images,labels);

  return 0;
}
