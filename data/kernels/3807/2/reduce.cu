#include "includes.h"
__global__ void reduce(float* d_out, float* d_in) { // Parallel summation: steps = O(log(N)), work = O(N * log(N))
extern __shared__ float sdata[];

int globId = blockDim.x * blockIdx.x + threadIdx.x;
int tid = threadIdx.x;

sdata[tid] = d_in[globId];
__syncthreads();

int s = blockDim.x >> 1;
while (s > 0) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
s >>= 1;
}
if (tid == 0) {
d_out[blockIdx.x] = sdata[0];
}
}