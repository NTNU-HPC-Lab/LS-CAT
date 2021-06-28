#include "includes.h"
__global__ void inputKernel(float *x, int n, int N)
{
int ix   = blockIdx.x * blockDim.x + threadIdx.x,i;
int iy   = blockIdx.y * blockDim.y + threadIdx.y;
int idx = iy * NUM_OF_X_THREADS + ix;

if (idx < N)
{
if (idx < n)
{
x[idx*N]  = (float)idx;
}
else
{
x[idx] = 0;
}

for(i=1;i<N;i++)
{
x[idx*N + i] = 0;
}
}

}