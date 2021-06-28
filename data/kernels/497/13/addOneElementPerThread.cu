#include "includes.h"
__global__ void addOneElementPerThread(double* a, double* b, double* c, int n)
{
// Get our global thread ID
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
int id = idy * n + idx;
// Make sure we do not go out of bounds
if (idx < n && idy < n)
c[id] = a[id] + b[id];
}