#include "includes.h"
// CUDA runtime

// helper functions and utilities to work with CUDA


extern "C"
__global__ void kernel(int* data, int size)
{
int id = blockDim.x * blockIdx.x + threadIdx.x;
data[id] = id;
}