#include "includes.h"
__global__ void ker_gkylCartFieldScale(unsigned s, unsigned nv, double fact, double *out)
{
for (int n = blockIdx.x*blockDim.x + threadIdx.x + s; n < s + nv; n += blockDim.x * gridDim.x)
out[n] *= fact;
}