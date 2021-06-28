#include "includes.h"
__global__ void DivideKernel ( float *d_dst, unsigned short *d_denom ) {
const int idx = blockIdx.x;
d_dst[idx] /= d_denom[idx];
}