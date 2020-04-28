#include <gtest/gtest.h>
#include <assert.h>
#include "KdV.h"

/**
* Test cases for index_looper
*
* Author: James "Andy" Edmond
* Written: March 29, 2020
*/

// Standard non-looping test
TEST(KdV_Testing,NonLoopingIndices) {
  int index = 2;
  int vec_size = 5;
  EXPECT_EQ(index_looper(2,5),2);
}

// Looping backwards
TEST(KdV_Testing,BackwardsLooping) {
  int index = -1;
  int vec_size = 5;
  EXPECT_EQ(index_looper(index,vec_size),3);
}

// Looping forwards
TEST(KdV_Testing,ForwardsLooping) {
  int index = 5;
  int vec_size = 5;
  EXPECT_EQ(index_looper(index,vec_size),1);
}
