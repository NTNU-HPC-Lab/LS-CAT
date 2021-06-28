#include "includes.h"
__global__ void SweHInit(double *var_in1, double *var_in2, double *var_out, int size) {
// Get thread id
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < size) {
// Transfer data and memory and calculation
var_out[tid] = var_in1[tid] - var_in2[tid];

// Thread id update
tid += blockDim.x * gridDim.x;
}
}