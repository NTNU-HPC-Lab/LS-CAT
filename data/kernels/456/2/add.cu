#include "includes.h"
__global__ void add(float *a, float *b, float *c) {
int tid = blockIdx.x;
while(tid < N) {
c[tid] = a[tid] + b[tid];
tid += gridDim.x;
}
}