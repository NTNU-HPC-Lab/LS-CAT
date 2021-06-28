#include "includes.h"
__global__ void VecAddInt32(int32_t* in0, int32_t* in1, int32_t* out, int cnt)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < cnt) {
out[tid] = in0[tid] + in1[tid];
}
}