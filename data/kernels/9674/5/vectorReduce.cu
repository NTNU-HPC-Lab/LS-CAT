#include "includes.h"
__global__ void vectorReduce(const float *global_input_data, float *global_output_data, const int numElements)
{
__shared__ float sdata[10];
__shared__  int sindice[10];

int tid = threadIdx.x;
int i = blockIdx.x * (blockDim.x ) + threadIdx.x;
sdata[tid] = global_input_data[i];
sindice[tid] = tid;
__syncthreads();

for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

if (tid < s ) {
if (sdata[tid] > sdata[tid + s]) {
sdata[tid] = sdata[tid + s];
sindice[tid] = sindice[tid + s];
}
__syncthreads();
}
}

__syncthreads();

if (tid == 0) {
global_output_data[0] = sdata[0];

}

if (tid == 1) {
global_output_data[1] = sindice[0];

}

}