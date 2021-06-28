#include "includes.h"
__global__ void dev_const(float *px, float k) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
px[tid] = k;
}