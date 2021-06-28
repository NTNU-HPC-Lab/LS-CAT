#include "includes.h"
__global__ void oneMinusTanh(float* out, float* in, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;

if(id < size)
out[id] = 1 - in[id];
}