#include "includes.h"
__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
int blockID = blockIdx.x;
int threadID = threadIdx.x;
int blockOffset = blockID * n;

extern __shared__ int temp[];
temp[2 * threadID] = input[blockOffset + (2 * threadID)];
temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

int offset = 1;
for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
{
__syncthreads();
if (threadID < d)
{
int ai = offset * (2 * threadID + 1) - 1;
int bi = offset * (2 * threadID + 2) - 1;
temp[bi] += temp[ai];
}
offset *= 2;
}
__syncthreads();


if (threadID == 0) {
sums[blockID] = temp[n - 1];
temp[n - 1] = 0;
}

for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
{
offset >>= 1;
__syncthreads();
if (threadID < d)
{
int ai = offset * (2 * threadID + 1) - 1;
int bi = offset * (2 * threadID + 2) - 1;
int t = temp[ai];
temp[ai] = temp[bi];
temp[bi] += t;
}
}
__syncthreads();

output[blockOffset + (2 * threadID)] = temp[2 * threadID];
output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}