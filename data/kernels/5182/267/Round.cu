#include "includes.h"
__global__ void Round(float * A, float  *out, int size) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id < size) {
int t = (int)(out[id] + 0.5);  // can it be speeded up??
out[id] = (float)t;
}
}