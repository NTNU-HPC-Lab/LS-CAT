#include "includes.h"
__global__ void dotProduct_CUDA_double(double *sum, int size, double *vector1, double *vector2){
int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
if(idx < size){
sum[idx] = (vector2[idx]) * (vector1[idx]);
}
}