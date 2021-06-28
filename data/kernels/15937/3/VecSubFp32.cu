#include "includes.h"
__global__ void VecSubFp32(float* in0, float* in1, float* out, int cnt)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < cnt) {
out[tid] = in0[tid] - in1[tid];
}
}