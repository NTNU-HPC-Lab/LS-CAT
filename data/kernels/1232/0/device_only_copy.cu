#include "includes.h"



__global__ void device_only_copy(float* output, float* input, size_t total_size){
for(size_t i = blockIdx.x * blockDim.x + threadIdx.x;
i < total_size;
i += blockDim.x * gridDim.x){
output[i] = input[i];
}
__syncthreads();
}