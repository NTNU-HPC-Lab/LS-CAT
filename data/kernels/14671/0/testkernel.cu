#include "includes.h"
/*
* test.cpp
* (c) 2015
* Author: Jim Fan
* See below link for how to support C++11 in eclipse
* http://scrupulousabstractions.tumblr.com/post/36441490955/eclipse-mingw-builds
*/

#ifdef is_CUDA
#endif

__global__ void testkernel()
{
double p = threadIdx.x + 66;
for (int i = 0; i < 30000000; ++i)
p += i / p - std::sqrt(p);

printf("thread %d; block %d\n", threadIdx.x, blockIdx.x);
}