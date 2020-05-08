cmake_minimum_required(VERSION 3.0)

PROJECT(DeepLearning)

# Find OpenMP
find_package(OpenMP)

# Find GTest
find_package(GTest)
if (NOT GTEST_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           release-1.8.0
    )
  #FetchContent_MakeAvailable(googletest)
  FetchContent_GetProperties(googletest)
  if (NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
  
  add_library(GTest::GTest INTERFACE IMPORTED)
  target_include_directories(GTest::GTest INTERFACE "${googletest_SOURCE_DIR}/googletest/include")
  target_link_libraries(GTest::GTest INTERFACE gtest)
  
  add_library(GTest::Main INTERFACE IMPORTED)
  target_link_libraries(GTest::Main INTERFACE gtest_main)
endif()


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


# Compiles executable for importing data test
#add_executable(mnist_example source/DataReader/main.cpp)
#target_compile_features(mnist_example PRIVATE cxx_range_for)
###target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
#target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="/home/jae1018/Proj2/MNIST_DeepLearning/data_set")


# Include direcs for header files
include_directories(include)


# -- Create DNN exec --
add_library(DNN_lib source/DNN/DNN.cxx)
target_link_libraries(DNN_lib xtensor)
add_executable(DNN_exec source/DNN/DNN_Main.cxx)
IF (OpenMP_CXX_FOUND)
  target_link_libraries(DNN_exec PRIVATE DNN_lib OpenMP::OpenMP_CXX)
endif()
#target_link_libraries(DNN_exec PRIVATE DNN_lib)
target_compile_features(DNN_exec PRIVATE cxx_range_for)
target_compile_definitions(DNN_exec PRIVATE MNIST_DATA_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/data_set")


# -- Testing ForwardProp --
add_executable(test_forw_prop source/DNN/test_forward_propagate.cxx)
target_link_libraries(test_forw_prop DNN_lib GTest::GTest GTest::Main)


# -- Testing BackwardProp --
add_executable(test_back_prop source/DNN/test_backpropagate.cxx)
target_link_libraries(test_back_prop DNN_lib GTest::GTest GTest::Main)
