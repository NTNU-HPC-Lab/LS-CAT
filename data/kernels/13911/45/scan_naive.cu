#include "includes.h"
__global__ void scan_naive(float *g_odata, float *g_idata, int n)
{
// Dynamically allocated shared memory for scan kernels
extern  __shared__  float temp[];

int thid = threadIdx.x;

int pout = 0;
int pin = 1;

// Cache the computational window in shared memory
temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;

for (int offset = 1; offset < n; offset *= 2)
{
pout = 1 - pout;
pin  = 1 - pout;
__syncthreads();

temp[pout*n+thid] = temp[pin*n+thid];

if (thid >= offset)
temp[pout*n+thid] += temp[pin*n+thid - offset];
}

__syncthreads();

g_odata[thid] = temp[pout*n+thid];
}