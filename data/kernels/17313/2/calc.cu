#include "includes.h"
__global__ void calc(float *result, float *b, float *a, int size){

int idx = blockIdx.x * blockDim.x + threadIdx.x;

if(idx < size){

float temp;

for (int j = 0; j < size; j++){
temp = *(a + j + (idx * size)) * (*(b + j));
atomicAdd(&result[idx], temp);
}
}
}