#include "includes.h"
__global__ void test(int* input,int* output) {
int index = threadIdx.x+blockIdx.x*blockDim.x +threadIdx.y+blockIdx.y*blockDim.y;
output[index] = input[index]*2;
}