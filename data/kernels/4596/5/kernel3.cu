#include "includes.h"
__global__ void kernel3(int* D, int* q, int b){

int i, j;

if(blockIdx.x >= b)
{
i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
}
else
{
i = blockIdx.x * blockDim.x + threadIdx.x;
}
if(blockIdx.y >= b)
{
j = (blockIdx.y + 1) * blockDim.y + threadIdx.y;
}
else
{
j = blockIdx.y * blockDim.y + threadIdx.y;
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