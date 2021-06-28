#include "includes.h"
__global__ void global_max( int *d_values, int *d_global_max ) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int val = d_values[tid];
atomicMax(d_global_max, val);
}