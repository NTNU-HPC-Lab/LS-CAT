#include "includes.h"
__global__ void multiplyKernel(float* Z, float* A, float* B, int size){
int id = blockDim.x * blockIdx.x + threadIdx.x;

if(id < size){
Z[id] = A[id] * B[id];
}
}