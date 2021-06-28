#include "includes.h"
__global__ void add_kernel_2elements(int* device_result, int* device_blocksum_2elements)
{
__shared__ int temp1;
int thid = threadIdx.x;
int N = blockDim.x;
if (thid == 0) temp1 = device_blocksum_2elements[blockIdx.x];
__syncthreads();
device_result[blockIdx.x * 4 * blockDim.x + thid] = device_result[blockIdx.x * 4 * blockDim.x + thid] + temp1;
device_result[blockIdx.x * 4 * blockDim.x + thid + N] =
device_result[blockIdx.x * 4 * blockDim.x + thid + N] + temp1;
device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] =
device_result[blockIdx.x * 4 * blockDim.x + thid + 2 * N] + temp1;
device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] =
device_result[blockIdx.x * 4 * blockDim.x + thid + 3 * N] + temp1;
}