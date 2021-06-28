#include "includes.h"
__global__ void kernel_5(float *d_data_in, float *d_data_out, int data_size)
{
__shared__ volatile float s_data[BLKSIZE];
int tid = threadIdx.x;
int index = tid + blockIdx.x*blockDim.x*2;
s_data[tid] = 0.0;
if (index < data_size){
s_data[tid] = d_data_in[index];
}
if ((index + blockDim.x) < data_size){
s_data[tid] += d_data_in[index + blockDim.x];
}
__syncthreads();

for (int s = blockDim.x / 2; s >= 64; s = s >> 1){
if (tid<s){
s_data[tid] += s_data[tid + s];
}
__syncthreads();
}

if (tid < 32){
s_data[tid] += s_data[tid + 32];
s_data[tid] += s_data[tid + 16];
s_data[tid] += s_data[tid + 8];
s_data[tid] += s_data[tid + 4];
s_data[tid] += s_data[tid + 2];
s_data[tid] += s_data[tid + 1];
}

if (tid == 0){
d_data_out[blockIdx.x] = s_data[tid];
}
}