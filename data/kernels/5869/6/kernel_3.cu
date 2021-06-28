#include "includes.h"
__global__ void kernel_3(float *d_data_in, float *d_data_out, int data_size)
{
__shared__ float s_data[BLKSIZE];
int tid = threadIdx.x;
int index = tid + blockIdx.x*blockDim.x;
s_data[tid] = 0.0;
if (index < data_size){
s_data[tid] = d_data_in[index];
}
__syncthreads();

for (int s = blockDim.x/2; s >= 1; s = s >> 1){
if (tid<s){
s_data[tid] += s_data[tid + s];
}
__syncthreads();
}

if (tid == 0){
d_data_out[blockIdx.x] = s_data[tid];
}
}