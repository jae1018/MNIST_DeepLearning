#include "DNN.h"

/**
* Note: Bias nodes are modified by setting a node_int val
* of -1 to the set_input and set_weight functions.
*
*
*
* STRUCTURE OF NETWORK:
*
* *** Input Layer (0) ***          *** Layer 1 ***
* ----------------------       -----------------------|
* | input has  | input |       | layer 1 has| Layer 1 |
* | no weights | nodes |       | weights    | Nodes   |
* |            |       | ----\ | for nodes  |         | -----\ ... until final
* |------------|-------| ----/ | from layer |         | -----/ ... layer.
* | and no bias nodes  |       | 0 to layer |         |
* | either             |       | 1 (called  |         |
* ---------------------        | w^(L=1)    |         |
*                              ------------------------
*                              | biases for each node |
*                              | activation in layer 1|
*                              | (called b^(L=1))     |
*                              ------------------------
*
* @author: James "Andy" Edmond
* @date: April 20, 2020 (420lol)
*/

// ********** Private Functions **********


// ----- Non-trivial Functions -----


// The activation function for the neural network
double DNN::activ_func(double val)  {
  return 1/(1 + exp(-val));
}


// The derivative of the activation function
double DNN::activ_func_deriv(double val) {
  return 1/(2 * (1 + cosh(val)) );
}


// Inverse of the activation function
// (Needed for going from activation to weighted sum)
double DNN::inv_activ_func(double val) {
  return -1 * log( (1/val) - 1 );
}


// The cost function used for evaluating error
// Note that technically, the cost function is evaluated over the sum of vector elements! But
// the result defined below is desired for computation.
double DNN::cost_func(double guess, double answer) {
  return pow((answer - guess),2) / 2;
}


// Derivative of the cost function
// (Change this when modifying cost function!)
double DNN::cost_func_deriv(double guess, double answer) {
  return (guess - answer);
}


// Forward propagates data from input layer all the way to output layer
void DNN::forward_propagate(vec& input) {
  set_activations(0,input);
  for (int layer_num = 1; layer_num < NUM_LAYERS; layer_num++) {

    // get data for calculations
    vec prev_activ_data = get_activations(layer_num - 1);  // activ data for layer L - 1
    vec next_activ_data = get_activations(layer_num);  // activ data for layer L
    arr weights = get_weights(layer_num);  // weights between layers L - 1 and L
    vec biases = get_biases(layer_num);  // biases for layer L
    int num_recv = get_num_recv_nodes(weights);

    // forward propagate to next layer
    for (int recv_node = 0; recv_node < num_recv; recv_node++) {
      vec weights_slice = xt::col(weights,recv_node);
      double weighted_sum = vdot(weights_slice,prev_activ_data) + biases(recv_node);
      next_activ_data(recv_node) = activ_func(weighted_sum);
    }
    set_activations(layer_num,next_activ_data);

  }
}


// Calculate the cost associated with the accurary of the estimated answer
// Return true if prediction is correct to within class constant TOLERANCE
bool DNN::analyze_output(vec& answers) {
  int last_layer_index = NUM_LAYERS - 1;
  vec last_activs = get_activations(last_layer_index);
  vec error_vec = xt::zeros<double>({LAYER_SIZES[last_layer_index]});
  for (int i = 0; i < LAYER_SIZES[last_layer_index]; i++) {
    double from_cost_deriv = cost_func_deriv(last_activs(i),answers(i));
    double from_activ_func_deriv = activ_func_deriv(inv_activ_func(last_activs(i)));
    error_vec(i) = from_cost_deriv * from_activ_func_deriv;
    //std::cout << "Node " << i << " has error " << error_vec(i) << "\n";
  }
  // Compute total cost and see if answer is correct
  assert( answers.size() == last_activs.size() );
  double cost = 0.0;
  bool return_val = true;
  for (int i = 0; i < answers.size(); i++) {
    cost += (1.0/2) * pow(answers(i) - last_activs(i),2);
    if (abs(answers(i) - last_activs(i)) > TOLERANCE) { return_val = false; }
  }
  std::cout << "Test yields cost of " << cost << std::endl;
  return return_val;
}


// Backward propagates based on correct_answer vec (which contains the correct answer to the
// current trial). Starts by computing error associated with cost function for final layer
// and propagates the error backward. After all error is determined, weights and biases
// are updated.
void DNN::backpropagate(vec& answers) {

  // Compute error of final layer
  //std::cout << "*** Computing error for last layer ***\n";
  //std::cout << "From answers = " << answers << " errors are determined\n";
  int last_layer_index = NUM_LAYERS - 1;
  vec last_activs = get_activations(last_layer_index);
  vec error_vec = xt::zeros<double>({LAYER_SIZES[last_layer_index]});
  for (int i = 0; i < LAYER_SIZES[last_layer_index]; i++) {
    double from_cost_deriv = cost_func_deriv(last_activs(i),answers(i));
    double from_activ_func_deriv = activ_func_deriv(inv_activ_func(last_activs(i)));
    error_vec(i) = from_cost_deriv * from_activ_func_deriv;
    //std::cout << "Node " << i << " has error " << error_vec(i) << "\n";
  }

  // Start saving errors for updating later
  // (only 4 vecs needed since no weights on input layer)
  vec errors[NUM_LAYERS - 1];
  errors[last_layer_index - 1] = error_vec;

  // Compute errors from semi-final layer to input layer
  for (int i = last_layer_index - 1; i > 0; i--) {
    //std::cout << "\n*** Computing error for layer " << i + 1 << " ***\n";
    arr weights = get_weights(i + 1);
    int num_send_nodes = get_num_send_nodes(weights);
    vec upper_layer_activs = get_activations(i);
    vec lower_layer_errors = xt::zeros<double>({num_send_nodes});
    vec upper_layer_errors = errors[i];
    // Compute error for individual neuron
    for (int q = 0; q < num_send_nodes; q++) {
      vec weight_slice = xt::row(weights,q);
      double weight_and_error_dot = vdot(weight_slice,upper_layer_errors);
      double from_cost_func_deriv = activ_func_deriv(  inv_activ_func(  upper_layer_activs(q)  )  );
      lower_layer_errors(q) = weight_and_error_dot * from_cost_func_deriv;
      //std::cout << "Node " << q << " has error " << lower_layer_errors(q) << "\n";
    }
    errors[i - 1] = lower_layer_errors;
  }

  // Update weights and biases
  // across each layer ...
  for (int i = 1; i < NUM_LAYERS; i++) {
    arr old_weights = get_weights(i);
    arr new_weights = xt::zeros<double>({ (old_weights.shape())[0], (old_weights.shape())[1] });
    vec old_biases = get_biases(i);
    vec new_biases = xt::zeros<double>({old_biases.size()});
    vec prev_layer_activs = get_activations(i - 1);
    vec errors_for_layer = errors[i - 1]; // --> means errors for layer i
    // ... and across each receiving node ...
    for (int recv = 0; recv < get_num_recv_nodes(old_weights); recv++) {
      // ... update the weights for each connection
      for (int send = 0; send < get_num_send_nodes(old_weights); send++) {
        double change = LEARNING_RATE * errors_for_layer(recv) * prev_layer_activs(send);
        // should it be plus???
        new_weights(send,recv) = old_weights(send,recv) - change;
      }
      // ... and update the bias for each receiving node
      new_biases(recv) = old_biases(recv) - LEARNING_RATE * errors_for_layer(recv);
    }
    set_weights(i,new_weights);
    set_biases(i,new_biases);
  }

}


// Save all weight and bias data to files
void DNN::save_data() {
  // for each layer ...
  for (int i = 1; i < NUM_LAYERS; i++) {
    // save the weights
    std::ofstream out_file_weights(data_folder_path + "weights_ " + std::to_string(i) + ".csv");
    xt::dump_csv(out_file_weights, get_weights(i));
    // and the biases
    std::ofstream out_file_biases(data_folder_path + "biases_" + std::to_string(i) + ".csv");
    xt::dump_csv(out_file_biases, get_biases(i));
  }
}


// Fill up all entries in weights arr
void DNN::initialize_weights(arr& weights) {
  int num_recv = get_num_recv_nodes(weights);
  int num_send = get_num_send_nodes(weights);
  for (int recv = 0; recv < num_recv; recv++) {
    for (int send = 0; send < num_send; send++) {
      //weights(send,recv) = double(recv*10 + send)/100;
      weights(send,recv) = rand()/RAND_MAX;
    }
  }
}


// Fill up all entires in biases vec
void DNN::initialize_biases(vec& biases) {
  for (int i = 0; i < biases.size(); i++) {
    //biases(i) = double(i)/100;
    biases(i) = rand()/RAND_MAX;
  }
}


// Starts nodes out from scratch (no prior weights saved to file imported here!!)
void DNN::initialize_network() {
  // Init weights and biases (incorpoate file-checking for prev vals later!)
  for (int i = 0; i < NUM_LAYERS-1; i++) {
    // Make weights matrix between adjacent layers and check
    // to see if they're already saved to files.
    std::ifstream read_data_weights;
    read_data_weights.open(data_folder_path + "weights_" + std::to_string(i) + ".csv");
    arr weights;
    if ( read_data_weights.good() ) {
      weights = xt::load_csv<double>(read_data_weights);
    } else { // make from scratch
      weights = xt::empty<double>({LAYER_SIZES[i],LAYER_SIZES[i+1]});
      initialize_weights(weights);
    }
    all_weights[i] = weights;
    // Make biases vector for all non-input layers and check
    // to see if they're already saved to files.
    std::ifstream read_data_biases;
    read_data_biases.open(data_folder_path + "biases_ " + std::to_string(i) + ".csv");
    vec biases;
    if ( read_data_biases.good() ) {
      biases = xt::load_csv<double>(read_data_biases);
    } else {
      biases = xt::empty<double>({LAYER_SIZES[i+1]});
      initialize_biases(biases);
    }
    all_biases[i] = biases;
  }
  // Init activations
  for (int i = 0; i < NUM_LAYERS; i++) {
    vec activations = xt::zeros<double>({LAYER_SIZES[i]});
    all_activations[i] = activations;
  }
}


// ----- Public Constructor -----


DNN::DNN(std::string path_to_saves) {
  data_folder_path = path_to_saves + "/";
  initialize_network();
}


// ----- Public Functions -----


// ----- Getters & Setters -----
// For all of these, layer_num is an int describing the index of what layer is being
// considered. 0 <= layer_num <= (NUM_LAYERS - 1) always!
// So if data for layer 4 (involving layers 1,2,3,4,5) is desired, then layer_num = 3.

// Based on ref to weight matrix, get number of receiving nodes
int DNN::get_num_recv_nodes(arr& weights) {
  return (  weights.shape()  )[1];
}


// Based on ref to weight matrix, get number of sending nodes
int DNN::get_num_send_nodes(arr& weights) {
  return (  weights.shape()  )[0];
}


// Returns weights for designated layer
// Note that input layer has no weights, so get_weights(0) throws error!
arr& DNN::get_weights(int layer_num) {
  return all_weights[layer_num-1];
}


// Sets the weights of the designated layer to the arr provided
void DNN::set_weights(int layer_num, arr& weights_in) {
  double rows = ((all_weights[layer_num - 1]).shape())[0];
  double cols = ((all_weights[layer_num - 1]).shape())[1];
  assert(  rows == (weights_in.shape())[0]  );
  assert(  cols == (weights_in.shape())[1]  );
  for (int i = 0; i < rows; i++) {
    for (int a = 0; a < cols; a++) {
      (all_weights[layer_num - 1])(i,a) = weights_in(i,a);
    }
  }
}


// Returns biases for designated layer
// Note that input layer has no biases, so get_biases(0) throws error!
vec& DNN::get_biases(int layer_num) {
  return all_biases[layer_num - 1];
}


// Sets the biases of the designated layer to the vec provided
// Note that all_biases[0] --> Layer 2 biases and ..[1] --> Layer 3 biases, etc.
void DNN::set_biases(int layer_num, vec& biases_in) {
  assert( (all_biases[layer_num - 1]).size() == biases_in.size() );
  for (int i = 0; i < biases_in.size(); i++) {
    (all_biases[layer_num - 1])(i) = biases_in(i);
  }
}


// Returns activations for designated layer
// The input values for the input layer are considered to be activations!
vec& DNN::get_activations(int layer_num) {
  return all_activations[layer_num];
}


// Sets the activations of the designated layer to the vec provided
void DNN::set_activations(int layer_num, vec& activ_in) {
  assert( LAYER_SIZES[layer_num] == activ_in.size() );
  for (int i = 0; i < activ_in.size(); i++) {
    (all_activations[layer_num])(i) = activ_in(i);
  }
}


// Returns the number of layers in the DNN
int DNN::get_num_layers() {
  return NUM_LAYERS;
}


// Returns the layer sizes in the DNN as an xtens
xt::xtensor<int,1> DNN::get_layer_sizes() {
  xt::xtensor<int,1> sizes = xt::empty<int>({NUM_LAYERS});
  for (int i = 0; i < NUM_LAYERS; i++) {
    sizes(i) = LAYER_SIZES[i];
  }
  return sizes;
}


// Trains the network over the dataset provided
void DNN::train_network(xt::xtensor<vec,1> images_in, vec labels_in) {
  assert( images_in.size() == labels_in.size() );
  int num_correct = 0;

  int max_iters = 500;//images_in.size();//100
  for (int test_num = 0; test_num < max_iters; test_num++) {
    // If new epoch is reach, save data
    if ( (test_num % EPOCH) == 0 ) { save_data(); }
    // Then calculate as usual
    std::cout << "On test " << test_num << "/" << max_iters << " ..." << std::endl;
    //set_activations(0,images_in(test_num));
    forward_propagate(images_in(test_num));
    // Make answer vector from label
    vec answer = xt::empty<double>({10});
    for (int q = 0; q < 10; q++) {
      answer(q) = (labels_in(test_num) == q) ? 1 : 0;  // if index == digit, then 1; else 0
    }
    if ( analyze_output(answer) ) { num_correct++; }
    std::cout << "For test num " << test_num << " ..." << std::endl
              << "guess = " << get_activations(NUM_LAYERS-1) << std::endl
              << "answer = " << answer << std::endl;
    backpropagate(answer);
  }
  std::cout << "Accuracy = " << double(num_correct)/max_iters << std::endl;
}

// Prints all the weights for all nodes
void DNN::print_all_weights() {
  std::cout << "\n----- printing weights -----\n"
            << "*** Layer 1 is input layer (so no weights or biases) ***\n";
  for (int i = 1; i < NUM_LAYERS; i++) {
    std::cout << "\n*** Layer " << i + 1 << " ***\n";
    arr weights = get_weights(i);
    vec biases = get_biases(i);
    int num_recv = get_num_recv_nodes(weights);
    int num_send = get_num_send_nodes(weights);
    for (int r_node = 0; r_node < num_recv; r_node++) { // r = receiving
      std::cout << "Node " << r_node + 1 << " has the following weights:\n[ ";
      for (int s_node = 0; s_node < num_send; s_node++) { // s = sending
        std::cout << weights(s_node,r_node) << " ";
      }
      std::cout << "] with bias = " << biases(r_node) << "\n";
    }
  }
}


// Test of forward propogate
void DNN::compute_forward() {
  //read_input();
  std::cout << "\n----- forward propagating -----\n";
  vec attempt = xt::empty<double>({LAYER_SIZES[0]});
  attempt.fill(0.5);
  forward_propagate(attempt);
  std::cout << "Output is " << get_activations(NUM_LAYERS - 1) << "\n";
}


// Test of backpropagate
void DNN::compute_backward() {
  vec answers = {0., 1.};
  std::cout << "\n----- backpropagating -----\n";
  backpropagate(answers);
}
