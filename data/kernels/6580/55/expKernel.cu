#include "includes.h"
__global__ void expKernel(float* Z, float* A, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;

if(id < size){
A[id] = __expf(-Z[id]);
}
}