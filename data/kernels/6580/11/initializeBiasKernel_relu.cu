#include "includes.h"
__global__ void initializeBiasKernel_relu(float* b, int size){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if(index < size){
b[index] = 0.0;
}
}