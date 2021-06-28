#include "includes.h"
__device__ float sigmoid(float x) {
return 1.0f / (1 + __expf(-x));
}
__global__ void initializeBiasKernel_sigmoid(float* b, int size){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if(index < size){
b[index] = 0.0;
}
}