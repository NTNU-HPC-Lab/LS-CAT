#include "includes.h"
__global__ void minValue(int *source, int *val){
__shared__ int temp[1];

int currentValue = source[threadIdx.x];
if (currentValue > -1 && currentValue < *val){
temp[0] = currentValue;
}

__syncthreads();

*val = temp[0];
}