#include "includes.h"
__global__ void ker_gkylCartFieldAssignAll(unsigned s, unsigned nv, double val, double *out)
{
for (int n = blockIdx.x*blockDim.x + threadIdx.x + s; n < s + nv; n += blockDim.x * gridDim.x)
out[n] = val;
}