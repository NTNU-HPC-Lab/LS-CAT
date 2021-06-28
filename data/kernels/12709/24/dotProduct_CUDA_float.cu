#include "includes.h"
__global__ void dotProduct_CUDA_float(float *sum, int size, float *vector1, float *vector2){
int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
if(idx < size){
sum[idx] = (vector2[idx]) * (vector1[idx]);
}
}