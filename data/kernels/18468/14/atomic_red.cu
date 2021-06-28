#include "includes.h"
__global__ void atomic_red(const float *gdata, float *out){
size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
if (idx < N) atomicAdd(out, gdata[idx]);
}