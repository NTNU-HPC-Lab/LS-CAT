#include "includes.h"

#define CUDA_KERNEL_LOOP(i ,n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

__global__ void add_bias_kernel( int n, float* data_out, const float* bias, const int out_channels, const int height_out, const int width_out ){
CUDA_KERNEL_LOOP(index, n){
const int c_col = (index / width_out / height_out) % out_channels;
float value = bias[c_col];
atomicAdd(data_out + index, value);
}
}