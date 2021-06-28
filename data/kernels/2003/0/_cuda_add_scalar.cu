#include "includes.h"

#define WARP_SIZE 32 // # of threads that are executed together (constant valid on most hardware)

/* Simple CUDA example showing:
1) how to sum the values of an array in parallel
2) how to add a scaler to values of an array in parallel
3) how to query GPU hardware

Compile with minimum archetecture specification of 30. Example:
nvcc example.cu - o example -arch=sm_30

Author: Jordan Bonilla
*/

// Allow timing of functions
clock_t start,end;

/* Add "scalar" to every element of the input array in parallel */

// CPU entry point for kernel to add "scalar" to every element of the input array
__global__ void _cuda_add_scalar(int *in, int scalar, int n)
{
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
while(globalIdx < n)
{
in[globalIdx] = in[globalIdx] + scalar;
globalIdx += blockDim.x * gridDim.x;
}
}