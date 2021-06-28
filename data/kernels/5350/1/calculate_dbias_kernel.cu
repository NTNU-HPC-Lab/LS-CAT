#include "includes.h"

#define CUDA_KERNEL_LOOP(i ,n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

__global__ void calculate_dbias_kernel( int n, const float* grad_output, float* grad_bias, const int out_channels, const int height_out, const int width_out ){
CUDA_KERNEL_LOOP(index, n){
const int c_col = (index / width_out / height_out) % out_channels;
float value = grad_output[index];
atomicAdd(grad_bias + c_col, value);
}
}