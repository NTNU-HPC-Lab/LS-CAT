#include "includes.h"
__global__ void kernel_1(float *d_data_in, float *d_data_out, int data_size)
{
__shared__ float s_data[BLKSIZE];
int tid = threadIdx.x;
int index = tid + blockIdx.x*blockDim.x;
s_data[tid] = 0.0;
if (index < data_size){
s_data[tid] = d_data_in[index];
}
__syncthreads();

for (int s = 2; s <= blockDim.x; s = s * 2){
if ((tid%s) == 0){
s_data[tid] += s_data[tid + s / 2];
}
__syncthreads();
}

if (tid == 0){
d_data_out[blockIdx.x] = s_data[tid];
}
}