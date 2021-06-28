#include "includes.h"
__global__ void add(float *loc, float *temp, const int num) {
int idx = blockIdx.x*blockDim.x+threadIdx.x;
if(idx < num) {
atomicAdd(loc,temp[idx]);
}
}