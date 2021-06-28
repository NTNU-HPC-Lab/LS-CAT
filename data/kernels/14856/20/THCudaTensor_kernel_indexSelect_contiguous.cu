#include "includes.h"
__global__ void THCudaTensor_kernel_indexSelect_contiguous( float *tensor, float *src, long stride, float *index, long idxSize)
{
// In the typical case, each block of 128 threads handles a 4x128
// section of the output with each warp handling a single 1x128 row.
// The outer loops handle inputs larger than 4*65535 or strides larger
// than 128*65535.
const int VT = 4;
const int WARP_SIZE = 32;
const int MAX_DIM_SIZE = 65535;

for (int idx = blockIdx.x * blockDim.y + threadIdx.y; idx < idxSize; idx += blockDim.y * MAX_DIM_SIZE) {
for (int startIdx = threadIdx.x + blockIdx.y * VT*WARP_SIZE; startIdx < stride; startIdx += VT*WARP_SIZE*MAX_DIM_SIZE) {
const long srcIdx = ((long) index[idx] - 1) * stride;
const long targetIdx = idx * stride;

#pragma unroll
for (int i = 0; i < VT; i++) {
const int featureIdx = startIdx + i * WARP_SIZE;
if (featureIdx < stride) {
tensor[targetIdx + featureIdx] = src[srcIdx + featureIdx];
}
}
}
}
}