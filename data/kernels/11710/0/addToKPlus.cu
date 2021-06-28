#include "includes.h"
__global__ void addToKPlus(int msize, double* a,  double* b, double* c, double* d)
{
int tid = threadIdx.x; // + blockIdx.x * blockDim.x;
if (tid < msize) {
d[tid] = a[tid] + b[tid] + c[tid];
// tid += blockDim.x*gridDim.x;`
}
}