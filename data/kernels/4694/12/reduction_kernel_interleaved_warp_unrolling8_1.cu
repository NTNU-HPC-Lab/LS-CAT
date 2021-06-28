#include "includes.h"
__global__ void reduction_kernel_interleaved_warp_unrolling8_1(int * input, int * temp_array, int size)
{
int tid = threadIdx.x;

//element index for this thread
int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

//local data pointer
int * i_data = input + blockDim.x * blockIdx.x * 8;

if ((index + 7 * blockDim.x) < size)
{
int a1 = input[index];
int a2 = input[index + blockDim.x];
int a3 = input[index + 2 * blockDim.x];
int a4 = input[index + 3 * blockDim.x];
int a5 = input[index + 4 * blockDim.x];
int a6 = input[index + 5 * blockDim.x];
int a7 = input[index + 6 * blockDim.x];
int a8 = input[index + 7 * blockDim.x];

input[index] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
}

__syncthreads();

for (int offset = blockDim.x / 2; offset >= 64;
offset = offset / 2)
{
if (tid < offset)
{
i_data[tid] += i_data[tid + offset];
}
__syncthreads();
}

if (tid < 32)
{
volatile int * vsmem = i_data;
vsmem[tid] += vsmem[tid + 32];
vsmem[tid] += vsmem[tid + 16];
vsmem[tid] += vsmem[tid + 8];
vsmem[tid] += vsmem[tid + 4];
vsmem[tid] += vsmem[tid + 2];
vsmem[tid] += vsmem[tid + 1];
}

if (tid == 0)
{
temp_array[blockIdx.x] = i_data[0];
}
}