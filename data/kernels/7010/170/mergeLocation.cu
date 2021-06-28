#include "includes.h"
__global__ void mergeLocation(const short2* loc_, float* x, float* y, const int npoints, float scale)
{
const int ptidx = blockIdx.x * blockDim.x + threadIdx.x;

if (ptidx < npoints)
{
short2 loc = loc_[ptidx];

x[ptidx] = loc.x * scale;
y[ptidx] = loc.y * scale;
}
}