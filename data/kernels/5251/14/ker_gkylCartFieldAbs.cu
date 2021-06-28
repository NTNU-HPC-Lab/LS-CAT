#include "includes.h"
__global__ void ker_gkylCartFieldAbs(unsigned s, unsigned nv, double *out)
{
for (int n = blockIdx.x*blockDim.x + threadIdx.x + s; n < s + nv; n += blockDim.x * gridDim.x)
out[n] = fabs(out[n]);
}