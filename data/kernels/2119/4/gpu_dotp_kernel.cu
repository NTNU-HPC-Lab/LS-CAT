#include "includes.h"
__global__ void gpu_dotp_kernel(int size, float* vec1, float* vec2, float* res){

float cache = 0;
int i = blockIdx.x * blockDim.x + threadIdx.x;

if( i < size ){
cache = vec1[i]*vec2[i];
}

atomicAdd(res, cache);
}