#include "includes.h"
__global__ void mapKernel(float* out, int functionCode, float frange_start, float dx) {
int id  = blockIdx.x * blockDim.x + threadIdx.x;
float x = frange_start + id * dx;
float y;

switch (functionCode) {
case 0: y = cos(x); break;
case 1: y = tan(x); break;
default: y = sin(x); break;
}

out[2 * id + 0] = x;
out[2 * id + 1] = y;
}