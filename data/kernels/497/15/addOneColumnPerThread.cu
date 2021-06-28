#include "includes.h"
__global__ void addOneColumnPerThread(double* a, double* b, double* c, int n)
{
// Get the column for current thread
int column = (blockIdx.x * blockDim.x + threadIdx.x);

// Make sure we do not go out of bounds
if (column < n)
{
for (int i = 0; i < n; i++)
{
c[i * n + column] = a[i * n + column] + b[i * n + column];
}
}
}