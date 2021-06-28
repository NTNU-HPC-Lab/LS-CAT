#include "includes.h"
__global__ void big_add(int *a, int *b, int *c, unsigned int N){
int tid;
tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
while(tid < N){
c[tid] = a[tid] + b[tid];
tid += stride;
}
}