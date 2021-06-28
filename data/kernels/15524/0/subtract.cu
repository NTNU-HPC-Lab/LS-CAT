#include "includes.h"

__constant__ float *c_Kernel;

__global__ void subtract(float *d_dst, float*d_src_1, float* d_src_2, int len) {


int baseX = blockIdx.x * blockDim.x + threadIdx.x;

if (baseX < len)
{
d_dst[baseX] = d_src_1[baseX] - d_src_2[baseX];
}

}