#include "includes.h"
__global__ void add(float *cudaA, float *kernel, float *cudaResult)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
int gid = idy * N + idx;

__shared__ float blockData[BLOCK_SIZE + 2 * BLUR_SIZE][BLOCK_SIZE + 2 * BLUR_SIZE][3];

int x = idx - BLUR_SIZE;
int y = idy - BLUR_SIZE;

if(x >= 0 && y >= 0)
for(int k = 0; k < 3; k++)
blockData[threadIdx.x][threadIdx.y][k] = cudaA[(gid - BLUR_SIZE - BLUR_SIZE * N)*3 + k];
else
for(int k = 0; k < 3; k++)
blockData[threadIdx.x][threadIdx.y][k] = 0;

x = idx + BLUR_SIZE;
y = idy - BLUR_SIZE;

if(x < N && y >= 0)
for(int k = 0; k < 3; k++)
blockData[threadIdx.y][threadIdx.x + 2 * BLUR_SIZE][k] = cudaA[(gid + BLUR_SIZE - BLUR_SIZE * N)*3 + k];
else
for(int k = 0; k < 3; k++)
blockData[threadIdx.y][threadIdx.x + 2 * BLUR_SIZE][k] = 0;

x = idx - BLUR_SIZE;
y = idy + BLUR_SIZE;

if(x >= 0 && y < N)
for(int k = 0; k < 3; k++)
blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x][k] = cudaA[(gid - BLUR_SIZE + BLUR_SIZE * N)*3 + k];
else
for(int k = 0; k < 3; k++)
blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x][k] = 0;

x = idx + BLUR_SIZE;
y = idy + BLUR_SIZE;

if(x < N && y < N)
for(int k = 0; k < 3; k++)
blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x + 2 * BLUR_SIZE][k] = cudaA[(gid + BLUR_SIZE + BLUR_SIZE * N)*3 + k];
else
for(int k = 0; k < 3; k++)
blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x + 2 * BLUR_SIZE][k] = 0;

__syncthreads();
for(int k = 0; k < 3; k++)
{
for(int i = -BLUR_SIZE; i <= BLUR_SIZE; i++)
for(int j = -BLUR_SIZE; j <= BLUR_SIZE; j++)
{
cudaResult[gid * 3 + k] += blockData[threadIdx.y + BLUR_SIZE + i][threadIdx.x + BLUR_SIZE + j][k] * kernel[(BLUR_SIZE - i) * (2 * BLUR_SIZE + 1) + (BLUR_SIZE - j)];
}
}
}