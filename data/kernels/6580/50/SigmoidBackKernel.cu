#include "includes.h"
__global__ void SigmoidBackKernel(float* Z, float* dZ, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id < size){
float t = Z[id];
dZ[id] = dZ[id] * t * (1-t) ;
}
}