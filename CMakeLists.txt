cmake_minimum_required(VERSION 3.0)

PROJECT(mnist_example)

# find xtensor package
find_package(xtl QUIET)
if (NOT xtl_FOUND)
  include(FetchContent)
  FetchContent_Declare(xtl
    GIT_REPOSITORY    https://github.com/xtensor-stack/xtl.git
    )
  #FetchContent_MakeAvailable(xtl)
  FetchContent_GetProperties(xtl)
  if (NOT xtl_POPULATED)
    FetchContent_Populate(xtl)
    add_subdirectory(${xtl_SOURCE_DIR} ${xtl_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()
  
find_package(xtensor QUIET)
if (NOT xtensor_FOUND)
  include(FetchContent)
  FetchContent_Declare(xtensor
    GIT_REPOSITORY    https://github.com/xtensor-stack/xtensor.git
    )
  #FetchContent_MakeAvailable(xtensor)
  FetchContent_GetProperties(xtensor)
  if (NOT xtensor_POPULATED)
    FetchContent_Populate(xtensor)
    add_subdirectory(${xtensor_SOURCE_DIR} ${xtensor_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()


# Tests xcontainer
add_executable(xtens_of_xtens source/DNN/building_xtensors_of_xtensors.cxx)
target_link_libraries(xtens_of_xtens xtensor)


# Compiles executable for importing data test
add_executable(mnist_example source/DataReader/main.cpp)
target_compile_features(mnist_example PRIVATE cxx_range_for)


# Create DNN exec
add_executable(DNN_exec source/DNN/Node_Manager.cxx)
target_link_libraries(DNN_exec xtensor)


# Pass MNIST data directory to main.cpp
#target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="/home/jae1018/Proj2/MNIST_DeepLearning/data_set")

