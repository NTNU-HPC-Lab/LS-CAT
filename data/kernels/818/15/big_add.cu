#include "includes.h"
__global__ void big_add(int *a, int *b, int *c, unsigned int N){
// init thread id
int tid;
tid = blockIdx.x * blockDim.x + threadIdx.x;
// stride is for big arrays, i.e. bigger than threads we have
int stride = blockDim.x * gridDim.x;

// do the operations
while(tid < N){
c[tid] = a[tid] + b[tid];
tid += stride;
}
}