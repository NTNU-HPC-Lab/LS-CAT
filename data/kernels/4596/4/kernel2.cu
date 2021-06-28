#include "includes.h"
__global__ void kernel2(int* D, int* q, int b){

int i, j;
if(blockIdx.y == 0)
{
j = b * blockDim.y + threadIdx.y;
if(blockIdx.x >= b)
{
i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
}
else
{
i = blockIdx.x * blockDim.x + threadIdx.x;
}
}
else
{
i = b * blockDim.y + threadIdx.y;
if(blockIdx.x >= b)
{
j = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
}
else
{
j = blockIdx.x * blockDim.x + threadIdx.x;
}
}

float d, f, e;
for(int k = b * THR_PER_BL; k < (b + 1) * THR_PER_BL; k++)
{
d = D[i * N + j];
f = D[i * N + k];
e = D[k * N + j];

__syncthreads();

if(d > f + e)
{
D[i * N + j] = f + e;
q[i * N + j] = k;
}
}
}