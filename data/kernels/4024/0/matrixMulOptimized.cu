#include "includes.h"
#define size 1024
#define block_size 32




__global__ void matrixMulOptimized(int* a, int* b, int* c)
{



__shared__ float a_share[32][32];
__shared__ float b_share[32][32];

int n = 1024;
int row = blockDim.y*blockIdx.y + threadIdx.y;
int col = blockDim.x*blockIdx.x + threadIdx.x;

int local_c = 0;
for (int i = 0; i < 32; ++i)
{
a_share[threadIdx.y][threadIdx.x] = a[row*n + i*blockDim.y + threadIdx.x];
b_share[threadIdx.y][threadIdx.x] = b[(i*blockDim.x + blockIdx.y)*n + col];

__syncthreads();
for (int k = 0; k < 32; ++k)
{
local_c += a_share[threadIdx.y][k]*b_share[k][threadIdx.x];
}
__syncthreads();

}

c[row*n + col] = local_c;
}