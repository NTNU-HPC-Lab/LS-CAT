#include "includes.h"
__global__ void kernel_unpack_yuy2_y16_cuda(const uint8_t * src, uint8_t *dst, int superPixCount)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

if (i >= superPixCount)
return;

for (; i < superPixCount; i += stride) {

int idx = i * 4;

dst[idx] = 0;
dst[idx + 1] = src[idx + 0];
dst[idx + 2] = 0;
dst[idx + 3] = src[idx + 2];
}
}