#include "includes.h"
__global__ void ReluBackKernel(float* Z, float* dZ, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id < size){
if(Z[id] <= 0) dZ[id] = 0;
}
}