#include "includes.h"
__global__ void expPlus(float* out, float* in, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;

if(id < size)
out[id] = __expf(in[id]);
}