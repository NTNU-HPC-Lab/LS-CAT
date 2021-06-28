#include "includes.h"
__global__ void kernel(int* count_d, float* randomnums)
{
int i;
double x,y,z;
int tid = blockDim.x * blockIdx.x + threadIdx.x;
i = tid;
int xidx = 0, yidx = 0;

xidx = (i+i);
yidx = (xidx+1);

x = randomnums[xidx];
y = randomnums[yidx];
z = ((x*x)+(y*y));

if (z<=1)
count_d[tid] = 1;
else
count_d[tid] = 0;
}