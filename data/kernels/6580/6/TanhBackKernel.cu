#include "includes.h"
__global__ void TanhBackKernel(float* Z, float* dZ, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id < size){
float t = (Z[id]);
dZ[id] = dZ[id] * (1-t*t) ;
}
}