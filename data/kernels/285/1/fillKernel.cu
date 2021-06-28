#include "includes.h"
__global__ void fillKernel(int *a, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < n) a[tid] = tid;
}