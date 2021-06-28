#include "includes.h"

static __device__ float E = 2.718281828;




__global__ void multiplyElementKernel(float *src1, float *src2, float *dst, int block_size)
{
int di = blockIdx.x * block_size + threadIdx.x;
dst[di] = src1[di] * src2[di];
}