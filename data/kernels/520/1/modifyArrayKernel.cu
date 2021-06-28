#include "includes.h"
__global__ void modifyArrayKernel(int *val, int *arr){
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < 6 && arr[i] > -1)
arr[i] = arr[i] - *val;
}