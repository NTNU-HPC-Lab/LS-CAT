#include "includes.h"
__global__ void kernel(float* red, float* green, float* blue, unsigned long N){

int x = threadIdx.x + (blockIdx.x * blockDim.x);
int y = threadIdx.y + (blockIdx.y * blockDim.y);
unsigned long tid = x + (y * blockDim.x * gridDim.x);

if(tid < N){
red[tid] = .5;
blue[tid] = .7;
green[tid]= .2;
}
}