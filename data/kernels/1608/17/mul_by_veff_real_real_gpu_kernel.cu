#include "includes.h"
__global__ void mul_by_veff_real_real_gpu_kernel(int nr__, double* buf__, double const* veff__)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < nr__) {
buf__[i] *= veff__[i];
}
}