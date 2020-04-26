cmake_minimum_required(VERSION 3.0)

PROJECT(DeepLearning)

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

# Get xtensor linalg package
#find_package(xtensor-blas)
#if (NOT xtensor-blas_FOUND)
#  include(FetchContent)
#  FetchContent_Declare(xtensor-blas
#    GIT_REPOSITORY    https://github.com/xtensor-stack/xtensor-blas
#    )
#  FetchContent_GetProperties(xtensor-blas)
#  if (NOT xtensor-blas_POPULATED)
#    FetchContent_Populate(xtensor-blas)
#    add_subdirectory(${xtensor-blas_SOURCE_DIR} ${xtensor-blas_BINARY_DIR} EXCLUDE_FROM_ALL)
#  endif()
#endif()



# Compiles executable for importing data test
#add_executable(mnist_example source/DataReader/main.cpp)
#target_compile_features(mnist_example PRIVATE cxx_range_for)
###target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
#target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="/home/jae1018/Proj2/MNIST_DeepLearning/data_set")


# Include direcs for header files
include_directories(include)


# Create DNN exec
#add_executable(DNN_exec source/DNN/DNN.cxx)
#add_library(DNN_lib source/DNN/DNN.cxx source/DNN/Node.cxx)
add_library(DNN_lib source/DNN/DNN.cxx)
target_link_libraries(DNN_lib xtensor)
add_executable(DNN_exec source/DNN/DNN_Main.cxx)
target_link_libraries(DNN_exec DNN_lib)


# Create Node main exec
#add_executable(Node_main source/DNN/Node_older.cxx)
#target_link_libraries(Node_main xtensor)


# create better Node main exec
##add_library(node_lib source/DNN/Node.cxx)
##link_libraries(node_lib)
#add_executable(Node_main_alt source/DNN/Node_main.cxx)
#target_link_libraries(Node_main_alt xtensor)
##target_link_libraries(Node_main_alt source/DNN/Node.cxx)
#target_link_libraries(Node_main_alt DNN_lib)
