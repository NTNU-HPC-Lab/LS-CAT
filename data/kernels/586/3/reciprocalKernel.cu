#include "includes.h"
/*
============================================================================
Name        : SpikeSorting.cu
Author      : John
Version     :
Copyright   :
Description : CUDA compute reciprocals
============================================================================
*/


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* CUDA kernel that computes reciprocal values for a given vector
*/

/**
* Host function that copies the data and launches the work on GPU
*/
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
if (idx < vectorSize)
data[idx] = 1.0/data[idx];
}