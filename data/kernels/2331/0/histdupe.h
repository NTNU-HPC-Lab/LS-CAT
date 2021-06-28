#pragma once


// ifdef used to ignore false positive CUDA syntax errors
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#define KERNEL_ARGS(grid, block)
#else
#define KERNEL_ARGS(grid, block) <<< grid, block >>>
#endif


// Simple data type describing a similar pair
typedef struct {
    int id1;
    int id2;
    float similarity;
} Pair;


/*
Kernel method signature

Takes in device pointers for data array, result array, confidence array, and result count.
Also takes size of data set and max allowed number of results.
*/
__global__ void histDupeKernel(const float*, const float*, const float*, const float*, int*, int*, int*, int*, float*, int*, const int, const int, const int);
