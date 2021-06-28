#include "includes.h"
/*
* Example of how to use the mxGPUArray API in a MEX file.  This example shows
* how to write a MEX function that takes a gpuArray input and returns a
* gpuArray output, e.g. B=mexFunction(A).
*
* Copyright 2012 The MathWorks, Inc.
*/


#define DIVUP(m,n)		((m)/(n)+((m)%(n)>0))
int const threadsPerBlock = (sizeof(unsigned long long) * 8);

/*
* Device code
*/
__device__ inline float devIoU(float const * const a, float const * const b)
{
float left = max(a[0], b[0]), right = min(a[2], b[2]);
float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
float interS = width * height;
float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
return interS / (Sa + Sb - interS);
}
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thres, const float *dev_boxes, unsigned long long *dev_mask)
{
const int row_start = blockIdx.y, col_start = blockIdx.x;
const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock), col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

//if (row_start > col_start) return;

__shared__ float block_boxes[threadsPerBlock * 5];
if (threadIdx.x < col_size)
{
block_boxes[threadIdx.x * 5 + 0] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
block_boxes[threadIdx.x * 5 + 1] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
block_boxes[threadIdx.x * 5 + 2] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
block_boxes[threadIdx.x * 5 + 3] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
block_boxes[threadIdx.x * 5 + 4] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
}
__syncthreads();

if (threadIdx.x < row_size)
{
const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
const float *cur_box = dev_boxes + cur_box_idx * 5;
int i = 0;
unsigned long long t = 0;
int start = 0;
if (row_start == col_start) start = threadIdx.x + 1;
for (i = start; i < col_size; i++)
{
if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thres)
{
t |= 1ULL << i;
}
}
const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
dev_mask[cur_box_idx * col_blocks + col_start] = t;
}
}