#include "includes.h"
__global__ void ker_gkylCartFieldAssign(unsigned s, unsigned nv, double fact, const double *inp, double *out)
{
for (int n = blockIdx.x*blockDim.x + threadIdx.x + s; n < s + nv; n += blockDim.x * gridDim.x)
out[n] = fact*inp[n];
}