#include "includes.h"
__global__ void reduction(float *g_odata, float *g_idata)
{
// dynamically allocated shared memory

extern  __shared__  float temp[];

int tid = threadIdx.x + blockIdx.x*blockDim.x;
int k = threadIdx.x;

// first, each thread loads data into shared memory

temp[k] = g_idata[tid];

// next, we perform binary tree reduction
int d = blockDim.x; if (d % 2) temp[0] += temp[d - 1];
for (d >>= 1; d > 0; d >>= 1) {
__syncthreads();  // ensure previous step completed
if (k<d) { temp[k] += temp[k + d]; }
if (k == 0 && d % 2 == 1 && d != 1) { temp[0] += temp[d - 1]; }
//printf("middle result:d:%d  temp[%d]:%f\n",d,k,temp[k]);
}

// finally, first thread puts result into global memory

if (tid == blockIdx.x*blockDim.x) {
g_odata[blockIdx.x] = temp[0];
//printf("g[%d]:%f\n",blockIdx.x,g_odata[blockIdx.x]);
}
}