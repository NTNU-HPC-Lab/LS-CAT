#include "includes.h"
__global__ void ArraySum(float *array, float *sum){
int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < N){
atomicAdd(sum, array[index]);
}
}