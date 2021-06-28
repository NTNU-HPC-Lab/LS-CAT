#include "includes.h"
__global__ void prefixSum(float* arr,int step){

int bx = blockIdx.x;
int tx = threadIdx.x;

int BX = blockDim.x;

int i = bx*BX+tx;

if(i < step) return;

int temp = arr[i-step];
__syncthreads();
arr[i] += temp;
}