#include "includes.h"
__global__ void helper(float * output, float * blocksum, int len) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < len){
for (int j=0; j<i/blockDim.x; j++)
output[i] += blocksum[j];

}

}