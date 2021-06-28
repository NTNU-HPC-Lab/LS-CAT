#include "includes.h"
__global__ void scan_x(int *g_odata, int *g_idata, int n)
{
extern __shared__ int temp[];  // allocated on invocation
int thid = threadIdx.x;
int bid = blockIdx.x;
int bdim = blockDim.x;
int offset = 1;
temp[2*thid]   = g_idata[bid*bdim*2 + 2*thid]; // load input into shared memory
temp[2*thid+1] = g_idata[bid*bdim*2 + 2*thid+1];
for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
{
__syncthreads();
if (thid < d)
{
int ai = offset*(2*thid+1)-1;
int bi = offset*(2*thid+2)-1;
temp[bi] += temp[ai];
}
offset *= 2;
}
if (thid == 0) { temp[n - 1] = 0; } // clear the last element
for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
{
offset >>= 1;
__syncthreads();
if (thid < d)
{
int ai = offset*(2*thid+1)-1;
int bi = offset*(2*thid+2)-1;
int t = temp[ai];
temp[ai] = temp[bi];
temp[bi] += t;
}
}
__syncthreads();
g_odata[bid*bdim*2 + 2*thid] = temp[2*thid+1]; // write results to device memory
int second_ind = 2*thid+2;
if(second_ind == bdim*2 )
{
g_odata[bid*bdim*2 + 2*thid+1] = temp[2*thid+1] + g_idata[bid*bdim*2 + 2*thid+1];
}
else
{
g_odata[bid*bdim*2 + 2*thid+1] = temp[2*thid+2];
}
}