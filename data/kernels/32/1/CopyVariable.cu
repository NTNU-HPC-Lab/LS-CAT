#include "includes.h"
__global__ void CopyVariable(double *var_in, double *var_out, int size) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < size) {
// Transfer data and memory
var_out[tid] = var_in[tid];

// Update thread id if vector is long
tid += blockDim.x * gridDim.x;
}
}