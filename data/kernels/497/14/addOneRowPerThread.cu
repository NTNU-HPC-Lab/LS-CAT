#include "includes.h"
__global__ void addOneRowPerThread(double* a, double* b, double* c, int n)
{
// Get the row for current thread
int row = (blockIdx.y * blockDim.y + threadIdx.y);

// Make sure we do not go out of bounds
if (row < n)
{
int idx = row * n;
for (int i = 0; i < n; i++)
{
c[idx + i] = a[idx + i] + b[idx + i];
}
}
}