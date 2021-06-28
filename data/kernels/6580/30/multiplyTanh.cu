#include "includes.h"
__global__ void multiplyTanh(float* out, float* in1, float* in2, int size){
int id = blockIdx.x * blockDim.x + threadIdx.x;

if(id < size)
out[id] = in1[id] * in2[id];
}