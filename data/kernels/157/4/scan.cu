#include "includes.h"
__global__ void scan(float * input, float * output, int len) {
//@@ Modify the body of this function to complete the functionality of
//@@ the scan on the device
//@@ You may need multiple kernel calls; write your kernels before this
//@@ function and call them from here
__shared__ float scan_array[BLOCK_SIZE];
int global_id = threadIdx.x + blockDim.x * blockIdx.x;
if (global_id < len)
scan_array[threadIdx.x] = input[global_id];
else
scan_array[threadIdx.x] = 0;
__syncthreads();
int stride = 1;
while (stride < BLOCK_SIZE) {
int index = (threadIdx.x + 1) * stride * 2 - 1;
if (index < BLOCK_SIZE)
scan_array[index] += scan_array[index - stride];
stride = stride << 1;
__syncthreads();
}

for(int stride = BLOCK_SIZE >> 1; stride > 0; stride = stride >> 1) {
__syncthreads();
int index = (threadIdx.x + 1) * stride * 2 - 1;
if (index + stride < BLOCK_SIZE)
scan_array[index + stride] += scan_array[index];
}
__syncthreads();
if (global_id < len)
output[global_id] = scan_array[threadIdx.x];


if (global_id < BLOCK_SIZE) {
__syncthreads();
for (int block_idx = 1; block_idx <= (len / BLOCK_SIZE) ; ++block_idx) {
float offset = output[block_idx * BLOCK_SIZE - 1];
if ((threadIdx.x + block_idx * blockDim.x) < len)
output[threadIdx.x + block_idx * blockDim.x] += offset;
__syncthreads();
}
}
}