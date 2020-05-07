#include <iostream>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "DNN.h"

int main() {

  // Initialize with string of path to saved weight and bias data
  // If data not found, will just run from scratch
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

  // Change these to test_images and test_labels after training algorithm!
  // Validation (Test) Size --> 10K     &&&     Training Size --> 60K
  auto images_orig = dataset.training_images; //dataset.test_images;
  auto labels_orig = dataset.training_labels;  //dataset.test_labels;

  xt::xtensor<vec,1> images = normalize(images_orig);
  vec labels = make_double_vector(labels_orig);

  std::cout << "*** Set size = " << labels_orig.size() << std::endl;
  std::cout << "*** Each image has " << images(0).size() << " pixels." << std::endl;

  // Been struggling with getting the algorithm to learn quickly, so let's
  // feed it the same test repeatedly to see if it eventually adapts at all...
  auto images_copy = images;
  auto labels_copy = labels;
  int skip = 100;   // for a $(skip) number of tests, the answer is the same
  int index = 0;
  for (int i = 0; i < labels.size(); i++) {
    if ((i + 1) % skip == 0) { index++; }
    images_copy(i) = images(index);
    labels_copy(i) = labels(index);
  }

  int num_epochs = 10;
  for (int i = 0; i < num_epochs; i++) {
    node_man.train_network(images_copy,labels_copy);
  }

  return 0;
}
