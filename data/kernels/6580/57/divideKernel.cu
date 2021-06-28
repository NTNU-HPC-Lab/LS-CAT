#include "includes.h"
__global__ void divideKernel(float* A, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;

if(id < size){
A[id] = 1.0f / A[id];
}
}