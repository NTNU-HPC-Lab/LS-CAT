#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void computeSum(float *d_filteredImage, float *d_imageSumGrid, unsigned int n)
{
__shared__ float smem[SMEM_SIZE];
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
float localSum = 0;

if (idx + 3 * blockDim.x < n)
{
float a1 = d_filteredImage[idx];
float a2 = d_filteredImage[idx + blockDim.x];
float a3 = d_filteredImage[idx + 2 * blockDim.x];
float a4 = d_filteredImage[idx + 3 * blockDim.x];
localSum = a1 + a2 + a3 + a4;
}

smem[tid] = localSum;
__syncthreads();

if (blockDim.x >= 1024 && tid < 512)
smem[tid] += smem[tid + 512];
__syncthreads();
if (blockDim.x >= 512 && tid < 256)
smem[tid] += smem[tid + 256];
__syncthreads();
if (blockDim.x >= 256 && tid < 128)
smem[tid] += smem[tid + 128];
__syncthreads();
if (blockDim.x >= 128 && tid < 64)
smem[tid] += smem[tid + 64];
__syncthreads();

if (tid < 32)
{
volatile float *vsmem = smem;
vsmem[tid] += vsmem[tid + 32];
vsmem[tid] += vsmem[tid + 16];
vsmem[tid] += vsmem[tid + 8];
vsmem[tid] += vsmem[tid + 4];
vsmem[tid] += vsmem[tid + 2];
vsmem[tid] += vsmem[tid + 1];
}

if (tid == 0) d_imageSumGrid[blockIdx.x] = smem[0];
}